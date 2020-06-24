import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
woman_list = ['woman', 'women', 'girl', 'girls']
man_list = ['man', 'men', 'boy', 'boys']
neutral = ['person', 'people', 'human']
gender_word_list = ['woman', 'women', 'girl', 'girls', 'man', 'men', 'boy', 'boys', 'person', 'people', 'human']


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, save_path, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        #plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    
    plt.show()
    plt.savefig(os.path.join(save_path, image_path.split('/')[-1]))

def analyze_result(base_path, gaic_path, gaices_path):
    num_candidate = 0
    list = []
    with open(base_path) as file:
        base_json = json.load(file)

    with open(gaic_path) as file:
        gaic_json = json.load(file)

    with open(gaices_path) as file:
        gaices_json = json.load(file)
    for key, value in base_json.items():
        if value == 'W' and gaic_json[key] is not 'W' and gaices_json[key] is not 'W':
            num_candidate += 1
            list.append(key)
    print(num_candidate)
    return list


def synthesize_gender_map(cocoid, image_path, mask_path, seq, alphas, rev_word_map, save_path, gender):
    img = imread(image_path)
    img = imresize(img, (14*24, 14*24))
    hit = False
    sum = 0
    gender_inferred = -1
    gender_result = "N"

    mask = imread(os.path.join(mask_path, cocoid + '.png'))
    mask = imresize(mask, (14 * 24, 14 * 24))

    plt.imshow(img)

    words = [rev_word_map[ind] for ind in seq]
    print(words)
    has_people = False

    for i, word in enumerate(words):
        if word in gender_word_list:
            has_people = True
            if word in woman_list:
                gender_inferred = 0
            else:
                gender_inferred = 1
            if gender == gender_inferred:
                gender_result = 'C'
            else:
                gender_result = 'W'
            gender_map = alphas[i-1]
            #gender_map = skimage.transform.pyramid_expand(gender_map, upscale=24, sigma=8)
            gender_map = imresize(gender_map, (14 * 24, 14 * 24))
            break
    if has_people:
        ind = np.unravel_index(np.argmax(gender_map, axis=None), gender_map.shape)
        print(gender_map[ind[0], ind[1]])
        if mask[ind[0], ind[1], 2] > 100:
            hit = True
            print('hit')
        else:
            print('failed')

        mask = np.asarray(mask[:, :, 2] > 0, dtype=int)
        gender_map = gender_map/np.sum(gender_map)
        sum = np.sum(mask * gender_map)
        print(sum)
        #plt.imshow(gender_map, alpha=0.4)
        #plt.plot(ind[0], ind[1], 'ro')
        #heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        #plt.set_cmap('jet')
        plt.axis('off')

    #plt.subplot(122)
    #plt.imshow(mask)
    plt.savefig(os.path.join(save_path, gender_result + '_' + str(cocoid) + '_' + ' '.join(words) + '.png'), bbox_inches='tight')
    #plt.show()
    return hit, sum, gender_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', default='/Users/tangruixiang/Desktop/Picture1.png', help='path to image')
    parser.add_argument('--model', '-m',
                        default='/Volumes/rxtang/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar',
                        help='path to model')
    parser.add_argument('--word_map', '-wm', default='/Users/tangruixiang/Desktop/Fairness_Dataset/code/proposed_Model_github/Image_Caption/processed_data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                        help='path to word map JSON')
    parser.add_argument('--save_path', '-sp', default='/Volumes/rxtang/caption_fig/original', help='path to save caption image')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    candidate_list = analyze_result('/Volumes/rxtang/caption_fig/result_base.json',
                                    '/Volumes/rxtang/caption_fig/result_GAIC.json',
                                    '/Volumes/rxtang/caption_fig/result_GAICes.json')
    # Load model
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    result_dict = {}

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)

    hit_num = 0
    failed_num = 0


    '''
    #Load json_file (secret test)
    with open("Ksplit_gender_category.json") as json_file:
        data = json.load(json_file)
    secret_test = data['secret_test']
    for item in secret_test:
        cocoid = str(item['coco_id'])
        if cocoid in candidate_list:
            gender = item['gender']
            image_path = os.path.join('/Volumes/rxtang/COCO/val2014',
                                      'COCO_val2014_' + '0'*(12 - len(cocoid)) + cocoid + '.jpg')
            # Encode, decode with attention and beam search
            seq, alphas = caption_image_beam_search(encoder, decoder, image_path, word_map, args.beam_size)
            alphas = torch.FloatTensor(alphas)
            result, sum, gender_result = synthesize_gender_map(cocoid=cocoid, mask_path='/Volumes/rxtang/COCO/secret_test_masks',
                                           image_path=image_path, seq=seq, alphas=alphas,
                                           rev_word_map=rev_word_map, save_path=args.save_path,
                                           gender=gender)
            if result:
                hit_num += 1
                failed_num += 1
            result_dict[cocoid] = gender_result

    #print(hit_num/(failed_num+hit_num))
    #with open(os.path.join(args.save_path, 'result.json'), 'w') as outfile:
    #    json.dump(result_dict, outfile)



    # Visualize caption and attention of best sequence
    #visualize_att(args.img, seq, alphas, rev_word_map, args.save_path, args.smooth)
    '''