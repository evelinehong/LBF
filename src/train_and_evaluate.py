from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
from .diagnosis_multistep import ExprTree
import math
import torch
import torch.optim
import torch.nn.functional as f
import time
import numpy as np

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

def prefix_to_infix(formula, length=None):
    if length is not None:
        formula = formula[:length]
    stack = []
    #prev_op = None
    #PRIORITY = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 1, "**": 1}
    for ch in reversed(formula):
        if not ch in ["+", "-", "*", "/", "^", "**"]:
            stack.append(ch)
        else:
            a = stack.pop()
            b = stack.pop()
            #if prev_op and PRIORITY[prev_op] < PRIORITY[ch]:
            exp = '('+a+ch+b+')'
            #else:
            #    exp = a+ch+b
            stack.append(exp)
            prev_op = ch
    return stack[-1]

def find_fix(pred, gt, all_prob, sym_list, num_start, n_step):
    """
    preds: batch_size * expr len                 int - predicted ids
    res: batch_size                              float - labeled correct result
    probs: batch_size * expr len * classes       float - predicted all probabilities
    num_list: batch_size * list
    """
    try:
        gt = eval(gt)

        for i in range(len(pred)):
            if  any(char.isdigit() for char in pred[i]):
                pred[i] = eval(pred[i].replace("%", "/100"))
            if pred[i] == "^":
                pred[i] = "**"
        
        for i in range(len(sym_list)):
            if  any(char.isdigit() for char in sym_list[i]):
                sym_list[i] = eval(sym_list[i].replace("%", "/100"))
            if sym_list[i] == "^":
                sym_list[i] = "**"
    except:
        return []

    tokens = list(zip(pred, all_prob))
    etree = ExprTree(sym_list, num_start)
    etree.parse(tokens)
    fix = []

    if abs(etree.res()[0] - gt) <= 1e-5:
        fix = [sym_list.index(i) for i in pred]
        # print("No fix needed")
    else:
        output = etree.fix(gt, n_step=n_step)
        if output:
            fix = [sym_list.index(i) for i in output[0]]


            #     print(f"  Fix found: {''.join(old_str)} "
            #             f"=> {''.join(new_str)} = {gt}")
            #     print(f"  {output}")
            # print ("fix found")
            # print (gt)
            # print (pred)

    return fix

def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    
    for i in range(target.size(0)):
        if target[i] == unk:
            print (num_stack)
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str

def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)
    test = out_expression_list(test_res, output_lang, num_list)

    if len(num_stack) == 0 and test_res == test_tar:
        test_exp = prefix_to_infix(test)
        return True, True, test, test_tar, test_exp

    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    try:
        test_exp = prefix_to_infix(test)
    except:
        test_exp = None
    if test is None:
        return False, False, test, tar, test_exp
    if test == tar:
        return True, True, test, tar, test_exp
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar, test_exp
        else:
            return False, False, test, tar, test_exp
    except:
        return False, False, test, tar, test_exp


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0)


def train_attn(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
               generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
               use_teacher_forcing=1, beam_size=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2
    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss


def evaluate_attn(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                  beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0].all_output
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0].all_output


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out, attn):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)
        self.attn = copy_list(attn) 


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, num_ans, num_list, buffer_batch, buffer_batch_exp, epoch, mask_flag = False, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    gen_length1 = [2*len(i)-1 for i in num_list]
    gen_length2 = [2*len(i)+1 for i in num_list]
    gen_length3 = [2*len(i)+3 for i in num_list]
    # gen_length4 = [2*len(i)+5 for i in num_list]
    # gen_length5 = [max(2*len(i)-1,1) for i in num_list]
    gen_lengths = [gen_length1, gen_length2, gen_length3]

    ori_target_length = target_length.copy()
    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i
        num_mask.append([0] * len(generate_nums) + [0] * d + [1] * (max_num_size - d - len(generate_nums)))
    num_mask = torch.ByteTensor(num_mask)

    batch_size = len(input_length)

    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    # target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0) 

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    # node_stacks_init = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # embeddings_stacks_init = [[] for _ in range(batch_size)]
    # left_childs_init = [None for _ in range(batch_size)]

    nums_stack_batch_init = nums_stack_batch.copy()

    fix_target_list = []
    fix_index = []
    fix_target_length = []
    fix_input_length = []
    fix_found = [False for _ in range (batch_size)]
    #explore but not update

    for gen_length in gen_lengths:
        target_length = gen_length
        max_target_length = max(target_length)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        nums_stack_batch = nums_stack_batch_init.copy()

        generate_exps = torch.zeros((max_target_length, batch_size), dtype = torch.int)
        #generate_exps = torch.zeros((max_gen_length, batch_size), dtype = torch.int)
    
        generated = [[] for j in range (batch_size)]
        num_flags = [False for j in range (batch_size)]
        generated_ops = [0 for j in range (batch_size)]
        generated_nums = [0 for j in range (batch_size)]
        all_node_outputs_mask = []

        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings, current_attn = predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

            num_score2 = num_score
            op2 = op
            # print (num_score2)
            # target length rules
            for i in range (batch_size):
                if t >= target_length[i]:
                    op2[i,1:] = -1e10
                    num_score2[i,:] = -1e10 #fix_length
                else:
                    if generated_ops[i] >= (target_length[i] - 1 ) / 2:
                        op2[i,:] = -1e10 #number of ops cannot be greater than (target_length-1)/2
                    if generated_nums[i] == generated_ops[i] and t < target_length[i] - 1:
                        num_score2[i,:] = -1e10 #except the last postion, number of nums cannot be greater than number of ops
                    if t == 0 and target_length[i] > 2:
                        num_score2[i,:] = -1e10 #first cannot be number unless target_length equals to 1
                    if t == target_length[i] - 1:
                        op2[i,:] = -1e10 #last is a number
                    if mask_flag:
                        num_score2[i][:len(generate_nums)] = -1e10 #for the first iterations, do not generate 1 and 3.14
                    if t == 1 and target_length[i] == 5 and epoch < 5:
                        if random.random() > 0.7:
                            num_score2[i,:] = -1e2
                        else:
                            op2[i,:] = -1e2
            # print (num_score2)

            outputs = torch.cat((op2, num_score2), 1)
            out_score = nn.functional.log_softmax(torch.cat((op2, num_score2), dim=1), dim=1)

            all_node_outputs_mask.append(outputs)

            topv, topi = out_score.topk(1)
            topi = topi.squeeze()
            generate_exps[t] = topi

            topi_t, generate_input = generate_tree_input(topi.tolist(), outputs, nums_stack_batch, num_start, unk)

            if USE_CUDA:
                generate_input = generate_input.cuda()
            
            left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context) 
            
            del generate_input

            left_childs = []

            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                node_stacks, topi_t.tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    generated_ops[idx] += 1
                else:
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                    generated[idx].append(i - num_start)
                    num_flags[idx] = True
                    generated_nums[idx] += 1
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
                
        generate_exps = generate_exps.transpose(0,1)

        all_node_outputs_mask = torch.stack(all_node_outputs_mask, dim=1)  # B x S x N
        buffer_batch_new = buffer_batch.copy()
        buffer_batch_new_exp = buffer_batch_exp.copy()

        for idx, exp, num, num_stack, gt in zip(range(batch_size), generate_exps, num_list, nums_stack_batch, torch.LongTensor(target_batch)):
            if fix_found[idx] == True:
                continue
            generate_exp = out_expression_list(exp, output_lang, num)
            ground = out_expression_list(gt, output_lang, num, copy.deepcopy(num_stack))
            all_list = output_lang.index2word[: num_start + len(generate_nums)] + num
            probs = all_node_outputs_mask[idx].detach().cpu().numpy()
            probs = probs[:target_length[idx]]
            probs = probs[:, :num_start+len(generate_nums)+len(num)]

            # # print (num)
            # print (num_ans[idx])
            fix = find_fix(
                        generate_exp[:target_length[idx]],
                        num_ans[idx],
                        probs,
                        all_list,
                        num_start,
                        50)
            if len(fix):
                fix_found[idx] = True
                fix_exp = out_expression_list(fix, output_lang, num)
                fix_infix = prefix_to_infix(fix_exp, target_length[idx])
                # print ("\n")
                # print (ground)
                # print (generate_exp)
                # print (fix_exp)

            if len(fix) and not fix in buffer_batch_new[idx]:
                # fix_dict = {}
                # keep = True
                # for token in fix:
                #     if token in fix_dict.keys():
                #         fix_dict[token] += 1
                #         if fix_dict[token] > 3:
                #             keep = False
                #     else:
                #         fix_dict[token] = 1
                # keep2 = True
                # if keep:
                #     fix_exp = out_expression_list(fix, output_lang, num)
                #     fix_infix = prefix_to_infix(fix_exp, target_length[idx])
                #     for pos in range (0,len(fix_infix)):
                #         if fix_infix[pos] == "-":
                #             try:
                #                 if fix_infix[pos-1] == fix_infix[pos+1] and fix_infix[pos-2] == "(" and fix_infix[pos+2] == ")":
                #                     keep2 = False
                #             except:
                #                 pass                    
                #         if fix_infix[pos] == "/":
                #             try:
                #                 if fix_infix[pos+1] == "1":
                #                     keep2 = False
                #                 if fix_infix[pos-1] == fix_infix[pos+1] and fix_infix[pos-2] == "(" and fix_infix[pos+2] == ")":
                #                     keep2 = False
                #             except:
                #                 pass
                #     if keep2:
                #         try:
                #             y = eval(fix_infix)
                #             if y == eval(num_ans[idx]):
                #                 buffer_batch_new[idx].append(fix)
                #                 buffer_batch_new_exp[idx].append(fix_infix)
                #         except:
                #             pass
                fix_exp = out_expression_list(fix, output_lang, num)
                fix_infix = prefix_to_infix(fix_exp, target_length[idx])
                try:
                    y = eval(fix_infix)
                    if y == eval(num_ans[idx]):
                        buffer_batch_new[idx].append(fix)
                        buffer_batch_new_exp[idx].append(fix_infix)
                except:
                    pass
            
            for buffer_fix in buffer_batch_new[idx]:
                fix_target_list.append(buffer_fix)                                                                                                                                                                                                                     
                fix_index.append(idx)
                fix_target_length.append(len(buffer_fix))
                fix_input_length.append(input_length[idx])

                # fix_target_list.append(target_batch[idx][:ori_target_length[idx]])
                # fix_index.append(idx)
                # fix_target_length.append(ori_target_length[idx])
                # fix_input_length.append(input_length[idx])

            # if  not len(buffer_batch_new[idx]):
            #     fix_target_list.append(target_batch[idx][:target_length[idx]])
            #     fix_index.append(idx)
            
                #     fix_exp = out_expression_list(buffer_fix, output_lang, num)
                    #fix_infix = prefix_to_infix(fix_exp, target_length[idx])
                    #print (fix_infix)
                #     print (fix_exp)
                # print (ground)

            # if not mapo_bs:
            #     if len(fix):
            #         #fix_exp = out_expression_list(fix, output_lang, num)
            #         #fix_infix = prefix_to_infix(fix_exp, target_length[idx])
            #         # print (fix_infix)
            #         # print (fix_exp)
            #         #number_fix += 1
            #         fix_target[idx][:target_length[idx]] = torch.LongTensor(fix)
            #     else:
            #         fix_length[idx] = 0
            # print (generate)
            # gen_infix = prefix_to_infix (generate, target_length[idx])
            # print (gen_infix)
        # print (number_fix)


    #explore ends
    #update begins

    # assign batches for buffers:
    fix_input_length = np.array(fix_input_length)
    fix_target_list = np.array(fix_target_list)
    fix_index = np.array(fix_index)
    fix_target_length = np.array(fix_target_length)
    inds = np.argsort(-fix_input_length)
    fix_target_list = fix_target_list[inds].tolist()
    fix_index = fix_index[inds].tolist()
    fix_target_length = fix_target_length[inds].tolist()

    mapo_batch_size = 64

    if not len(fix_target_list) % mapo_batch_size == 0:
        num_iteration = int(len(fix_target_list)/mapo_batch_size) + 1 
    else:
        num_iteration = int(len(fix_target_list)/mapo_batch_size)

    for j in range (num_iteration):
        if not j * mapo_batch_size + mapo_batch_size - 1 < len(fix_target_list):
            mapo_batch_size = len(fix_target_list) - ((j - 1) * mapo_batch_size + mapo_batch_size)
        target_length_mapo = fix_target_length[j * mapo_batch_size : (j * mapo_batch_size + mapo_batch_size)]
        idx_list = fix_index[j * mapo_batch_size : (j * mapo_batch_size + mapo_batch_size)]
        target_list = fix_target_list[j * mapo_batch_size : (j * mapo_batch_size + mapo_batch_size)]
        input_length_mapo = []
        num_pos_mapo = []
        num_size_mapo_batch = []
        nums_stack_mapo_batch = []        
        max_length_batch = []
        for k in range (mapo_batch_size):
            idx = idx_list[k]           
            input_length_mapo.append(input_length[idx])
            num_pos_mapo.append(num_pos[idx])
            num_size_mapo_batch.append(num_size_batch[idx])
            nums_stack_mapo_batch.append(copy.deepcopy(nums_stack_batch[idx]))
            # max_length_batch.append(max(ori_target_length[idx], target_length_mapo[k]))

        input_var_mapo = torch.zeros((max(input_length_mapo), mapo_batch_size),dtype=torch.long)
        # target = torch.zeros((mapo_batch_size, max(target_length_mapo)), dtype = torch.long)
        target = torch.zeros((mapo_batch_size, max(target_length_mapo)), dtype = torch.long)
        # ori_target = torch.zeros((mapo_batch_size, max(max_length_batch)), dtype = torch.long)

        for k in range (mapo_batch_size):
            idx = idx_list[k]
            input_var_mapo[:,k] = input_var[:,idx][:max(input_length_mapo)]
            target[k][:target_length_mapo[k]] = torch.LongTensor(target_list[k])
            # ori_target[k][:ori_target_length[idx]] = torch.LongTensor(target_batch[idx][:ori_target_length[idx]])
        seq_mask = []
        max_len = max(input_length_mapo)
        for i in input_length_mapo:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.ByteTensor(seq_mask)

        unk = output_lang.word2index["UNK"]

        num_mask = []
        max_num_size = max(num_size_mapo_batch) + len(generate_nums)
        for i in num_size_mapo_batch:
            d = i
            num_mask.append([0] * len(generate_nums) + [0] * d + [1] * (max_num_size - d - len(generate_nums)))
        num_mask = torch.ByteTensor(num_mask)

        batch_size = len(input_length_mapo)

        target = torch.LongTensor(target).transpose(0, 1)
        # ori_target = torch.LongTensor(ori_target).transpose(0, 1)

        padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0) 

        encoder.train()
        predict.train()
        generate.train()
        merge.train()

        if USE_CUDA:
            input_var_mapo = input_var_mapo.cuda()
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()

        # Zero gradients of both optimizers
        encoder_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        generate_optimizer.zero_grad()
        merge_optimizer.zero_grad()
        # Run words through encoder

        encoder_outputs, problem_output = encoder(input_var_mapo, input_length_mapo)
        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        max_target_length = max(target_length_mapo)

        copy_num_len = [len(_) for _ in num_pos_mapo]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos_mapo, batch_size, num_size,
                                                                  encoder.hidden_size)
        num_start = output_lang.num_start
        embeddings_stacks = [[] for _ in range(batch_size)]

        left_childs = [None for _ in range(batch_size)]
    
        all_node_outputs = []

        nums_stack_mapo_batch_copy = copy.deepcopy(nums_stack_mapo_batch)
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings, current_attn = predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            # print (target[t])
            # print (ori_target[t])
            # print (nums_stack_mapo_batch)
            # print (nums_stack_batch)

            target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_mapo_batch, num_start, unk)
            # ori_target_t, ori_generate_input = generate_tree_input(ori_target[t].tolist(), outputs, nums_stack_mapo_batch_copy, num_start, unk)
            target[t] = target_t
            # ori_target[t] = ori_target_t
            if USE_CUDA:
                generate_input = generate_input.cuda()
            left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
        
        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
        target = target.transpose(0, 1).contiguous()
        # ori_target = ori_target.transpose(0, 1).contiguous()

        if USE_CUDA:
            # all_leafs = all_leafs.cuda()
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()
            # ori_target = ori_target.cuda()


        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
        loss1 = masked_cross_entropy(all_node_outputs, target, target_length_mapo)
        loss = loss1
        #loss0 = masked_cross_entropy2(all_node_outputs, ori_target, max_length_batch)
        #loss = 0.75 * loss0 + 0.25 * loss1
        #loss = loss1
        loss.backward()
        # clip the grad
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

        # Update parameters with optimizers
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()

    return loss.item(), buffer_batch_new, num_iteration, buffer_batch_new_exp  # , loss_0.item(), loss_1.item()
    # return 0


def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]

    current_attns = []
    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs
            current_attns = b.attn

            num_score, op, current_embeddings, current_context, current_nums_embeddings, current_attn = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            current_attns.append(torch.squeeze(current_attn).detach().cpu().numpy())

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)


            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                            current_left_childs, current_out, current_attns))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return [[beams[0].out, beams[0].attn], [beams[1].out, beams[1].attn], [beams[2].out, beams[2].attn], [beams[3].out, beams[3].attn], [beams[4].out, beams[4].attn]]


def topdown_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch,
                       generate_nums, encoder, predict, generate, encoder_optimizer, predict_optimizer,
                       generate_optimizer, output_lang, num_pos, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    left_childs = [None for _ in range(batch_size)]
    node_stacks_copy = node_stacks.copy()

    for t in range(max_target_length):
        node_stacks = node_stacks_copy
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        for idx, l, r, node_stack, i in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                            node_stacks, target[t].tolist()):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def topdown_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, output_lang, num_pos,
                          beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, embeddings_stacks, left_childs,
                                              current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
