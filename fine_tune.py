import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention, Fine_Tune_DecoderWithAttention, RemoveGenderRegion
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt

from scipy.misc import imread, imresize

# Data parameters
data_folder = '/Volumes/rxtang/Fairness_Dataset/code/Proposed_Model/processed_FT_data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 500  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 1
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
alpha_m = 0.5  # regularization parameter for mask attention
alpha_n = 0.1  # regularization parameter for remove gender region quality
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
freeze_decoder_lstm = True  # freeze decoder lstm?
supervised_training =True
save_example_image = True
checkpoint = '/Users/tangruixiang/Desktop/Fairness_Dataset/code/proposed_Model_github/Image_Caption/models/download/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # path to checkpoint, None if none
#checkpoint = None
checkpoint_savepath = 'models/exp4'

# environment parameter
is_cpu = True

word_map_file = '/Users/tangruixiang/Desktop/Fairness_Dataset/code/proposed_Model_github/Image_Caption/processed_FT_data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    decoder = Fine_Tune_DecoderWithAttention(attention_dim=attention_dim,
                                             embed_dim=emb_dim,
                                             decoder_dim=decoder_dim,
                                             vocab_size=len(word_map),
                                             dropout=dropout)

    val_decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None
    g_remover = RemoveGenderRegion()

    if checkpoint is not None:
        if is_cpu:
            checkpoint = torch.load(checkpoint,  map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']

        decoder = load_parameter(checkpoint['decoder'], decoder)
        encoder = load_parameter(checkpoint['encoder'], encoder)

        # decoder_optimizer = checkpoint['decoder_optimizer']
        decoder_optimizer = load_parameter(checkpoint['decoder_optimizer'], decoder_optimizer)
        #encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder_optimizer = load_parameter(checkpoint['encoder_optimizer'], encoder_optimizer)

        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
        if freeze_decoder_lstm:
            decoder.freeze_LSTM(freeze=True)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    g_remover = g_remover.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # fix CUDA bug
    if not is_cpu:
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    '''
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    '''

    if not supervised_training:
        train_loader = torch.utils.data.DataLoader(
            Fine_Tune_CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
            batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            Fine_Tune_CaptionDataset_With_Mask(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
            batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        if not supervised_training:
            # One epoch's training
            self_guided_fine_tune_train(train_loader=train_loader,
                                        encoder=encoder,
                                        decoder=decoder,
                                        criterion=criterion,
                                        encoder_optimizer=encoder_optimizer,
                                        decoder_optimizer=decoder_optimizer,
                                        g_remover=g_remover,
                                        epoch=epoch)
        else:
            supervised_guided_fine_tune_train(train_loader=train_loader,
                                              encoder=encoder,
                                              decoder=decoder,
                                              criterion=criterion,
                                              encoder_optimizer=encoder_optimizer,
                                              decoder_optimizer=decoder_optimizer,
                                              g_remover=g_remover,
                                              epoch=epoch)

        # One epoch's validation

        val_decoder = load_parameter(decoder, val_decoder)
        val_decoder = val_decoder.to(device)

        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=val_decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, checkpoint_savepath) 

def load_parameter(checkpoint, module):
    checkpoint = checkpoint.state_dict()
    model_dict = module.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    module.load_state_dict(model_dict)
    return module

def self_guided_fine_tune_train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, g_remover, epoch):
    """
    fine tune train for mitigate gender bias
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    #for i, (imgs, caps, caplens) in enumerate(train_loader):
    for i, (imgs, caps, caplens, n_caps, gender_pos) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        raw_imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        n_caps = n_caps.to(device)
        gender_pos = gender_pos.to(device)

        ############################################## First Forward propv #############################################
        imgs = encoder(raw_imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind, gender_alphas = decoder(imgs, caps, caplens, gender_pos)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        ############################################## Second Forward propv #############################################
        # process images : remove gender related regions
        n_imgs, soft_mask = g_remover(raw_imgs, gender_alphas)
        
        if save_example_image:
            index = 0
            # plot example images
            p_raw_imgs = (raw_imgs.data.cpu()).numpy().transpose((0, 2, 3, 1))
            p_n_imgs = (n_imgs.data.cpu()).numpy().transpose((0, 2, 3, 1))
            p_g_alphas = (soft_mask.data.cpu()).numpy().transpose((0, 2, 3, 1))
            p_caps = (caps.cpu()).numpy()[index]
            p_n_caps = (n_caps.cpu()).numpy()[index]

            p_caps = [rev_word_map[ind] for ind in p_caps]
            p_n_caps = [rev_word_map[ind] for ind in p_n_caps]
            p_caps = ' '.join(p_caps)
            p_n_caps = ' '.join(p_n_caps)

            plt.subplot(221)
            plt.imshow(p_raw_imgs[index, :])
            plt.subplot(222)
            plt.imshow(p_n_imgs[index, :])
            plt.subplot(223)
            plt.imshow(p_g_alphas[index, :, :, 0])
            plt.subplot(224)
            plt.text(0.05, 0.05, p_caps)
            plt.text(0.05, 0.5, p_n_caps)
            plt.show()
            plt.savefig('test.png')


        n_imgs = encoder(n_imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind, _ = decoder(n_imgs, n_caps, caplens, gender_pos)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        '''
        # Calculate loss
        loss += criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        '''
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def supervised_guided_fine_tune_train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, g_remover, epoch):
    """
    fine tune train for mitigate gender bias
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    #for i, (imgs, caps, caplens) in enumerate(train_loader):
    for i, (imgs, caps, caplens, n_caps, gender_pos, mask_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        raw_imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        n_caps = n_caps.to(device)
        gender_pos = gender_pos.to(device)
        mask_imgs = mask_imgs.to(device)

        ############################################## First Forward propv #############################################
        imgs = encoder(raw_imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind, gender_alphas = decoder(imgs, caps, caplens, gender_pos)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # add gender word loss
        gender_targets = targets[torch.arange(len(decode_lengths)), (gender_pos[sort_ind]-1).squeeze()]
        gender_scores = scores[torch.arange(len(decode_lengths)), (gender_pos[sort_ind]-1).squeeze()]
        gender_loss_1 = criterion(gender_scores, gender_targets)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        quality_loss_1 = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        d_att_loss_1 = alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        ############################################## Second Forward propv #############################################
        # process images : remove gender related regions
        n_imgs, soft_mask, g_alphas, r_mask_imgs = g_remover(raw_imgs, gender_alphas, mask_imgs)

        enc_n_imgs = encoder(n_imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind, _ = decoder(enc_n_imgs, n_caps, caplens, gender_pos)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # add gender word loss
        gender_targets = targets[torch.arange(len(decode_lengths)), (gender_pos[sort_ind] - 1).squeeze()]
        gender_scores = scores[torch.arange(len(decode_lengths)), (gender_pos[sort_ind] - 1).squeeze()]
        gender_loss_2 = criterion(gender_scores, gender_targets)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        quality_loss_2 = criterion(scores, targets)

        # Add supervised mask loss
        r_mask_imgs = r_mask_imgs[:, 0, :, :].unsqueeze(1)
        mask_loss = (len(decode_lengths) - (r_mask_imgs * g_alphas).sum())
        # mask_imgs = mask_imgs[:, 0, :, :].unsqueeze(1)
        # mask_loss = ((soft_mask * (1 - mask_imgs)) ** 2).sum()
        # loss += mask_loss

        # Add doubly stochastic attention regularization
        d_att_loss_2 = alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        loss = quality_loss_1 + gender_loss_1  + d_att_loss_1 * alpha_c\
               + quality_loss_2 * alpha_n + gender_loss_2 + d_att_loss_2 * alpha_c + mask_loss * alpha_m

        print("quality_loss_1:",quality_loss_1,"gender_loss_1", gender_loss_1, "d_att_loss_1", d_att_loss_1)
        print("quality_loss_2:",quality_loss_2,"gender_loss_2", gender_loss_2, "d_att_loss_2", d_att_loss_2)
        print("mask_loss:",mask_loss)
        # save fig
        if save_example_image:
            # plot example images
            index=0
            p_raw_imgs = (raw_imgs.data.cpu()).numpy().transpose((0, 2, 3, 1))
            p_n_imgs = (n_imgs.data.cpu()).numpy().transpose((0, 2, 3, 1))
            p_g_alphas = (soft_mask.data.cpu()).numpy().transpose((0, 2, 3, 1))
            p_g_mask_imgs = (mask_imgs.data.cpu()).numpy().transpose((0, 2, 3, 1))
            p_g_mask_loss_img = (soft_mask * (1 - mask_imgs))
            p_g_mask_loss_img = (p_g_mask_loss_img.data.cpu()).numpy().transpose((0, 2, 3, 1))
            g_alphas = (g_alphas.data.cpu()).numpy().transpose((0, 2, 3, 1))
            resize_mask = (r_mask_imgs.cpu()).numpy().transpose((0, 2, 3, 1))

            p_caps = (caps.cpu()).numpy()[index]
            p_n_caps = (n_caps.cpu()).numpy()[index]

            p_caps = [rev_word_map[ind] for ind in p_caps]
            p_n_caps = [rev_word_map[ind] for ind in p_n_caps]
            resize_mask = resize_mask[index, :, :, 0]

            plt.subplot(241)
            plt.imshow(p_raw_imgs[index, :])
            plt.subplot(242)
            plt.imshow(p_n_imgs[index, :])
            plt.subplot(243)
            plt.imshow(p_g_alphas[index, :, :, 0])
            plt.subplot(244)
            plt.imshow(p_g_mask_imgs[index, :, :, 0])
            plt.subplot(245)
            plt.imshow(p_g_mask_loss_img[index, :, :, 0])
            plt.subplot(246)
            plt.imshow(resize_mask)
            plt.subplot(247)
            plt.imshow(g_alphas[index, :, :, 0])
            plt.subplot(248)
            plt.text(0.05, 0.05, p_caps)
            plt.text(0.05, 0.5, p_n_caps)
            plt.show()
            plt.savefig('test.png')

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
