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

def generate_tree_input(target, decoder_output, num_start):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, num_start):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    
    return target



def compute_prefix_tree_result(test_res, ans, output_lang, num_list):
    # print(test_res, test_tar)
    test = out_expression_list(test_res, output_lang, num_list)
    try:
        test_exp = prefix_to_infix(test)
    except:
        test_exp = None
    if test is None:
        return False, test_exp
    try:
        if abs(compute_prefix_expression(test) - eval(ans)) < 1e-4:
            return True, test_exp
        else:
            return False, test_exp
    except:
        return False, test_exp

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
    return all_num.masked_fill_(masked_index, 0.0)

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
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def train_tree(input_batch, input_length, num_size_batch, 
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, num_ans, num_list, buffer_batch, buffer_batch_exp, epoch, model = 'fix', n_step = 50, mask_flag = False, english=False):
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
    # gen_length5 = [max(2*len(i)-3, 1) for i in num_list]
    gen_lengths = [gen_length1, gen_length2, gen_length3]

    num_mask = []
    max_num_size = max(num_size_batch) + 2
    for i in num_size_batch:
        d = i
        num_mask.append([0] * 2 + [0] * d + [1] * (max_num_size - d - 2))
    num_mask = torch.ByteTensor(num_mask)

    batch_size = len(input_length)

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

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

        generate_exps = torch.zeros((max_target_length, batch_size), dtype = torch.int)
    
        generated = [[] for j in range (batch_size)]
        num_flags = [False for j in range (batch_size)]
        generated_ops = [0 for j in range (batch_size)]
        generated_nums = [0 for j in range (batch_size)]
        all_node_outputs_mask = []


        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
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
                        num_score2[i][:2] = -1e10 #for the first iterations, do not generate 1 and 3.14
                    if t == 1 and target_length[i] == 5 and epoch < 5:
                        if random.random() > 0.7:
                            num_score2[i,:] = -1e2
                        else:
                            op2[i,:] = -1e2
                    if epoch < 5 and model in ['reinforce', 'mapo']:
                        for gen in generated[i]:
                            num_score2[i,gen] = -1e10
            # print (num_score2)

            outputs = torch.cat((op2, num_score2), 1)
            out_score = nn.functional.log_softmax(torch.cat((op2, num_score2), dim=1), dim=1)

            all_node_outputs_mask.append(outputs)

            topv, topi = out_score.topk(1)
            topi = topi.squeeze()
            generate_exps[t] = topi

            topi_t, generate_input = generate_tree_input(topi.tolist(), outputs, num_start)

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

        for idx, exp, num in zip(range(batch_size), generate_exps, num_list):
            if fix_found[idx] == True:
                continue
            generate_exp = out_expression_list(exp, output_lang, num)
            all_list = output_lang.index2word[: num_start + 2] + num
            probs = all_node_outputs_mask[idx].detach().cpu().numpy()
            probs = probs[:target_length[idx]]
            probs = probs[:, :num_start+2+len(num)]

            if model in ['fix', 'ma-fix']:
                fix = find_fix(
                            generate_exp[:target_length[idx]],
                            num_ans[idx],
                            probs,
                            all_list,
                            num_start,
                            n_step)

                if len(fix):
                    fix_found[idx] = True
                    fix_exp = out_expression_list(fix, output_lang, num)
                    fix_infix = prefix_to_infix(fix_exp, target_length[idx])
                    try:
                        y = eval(fix_infix)
                        #if y == eval(num_ans[idx]):
                        if model == 'fix':
                            fix_target_list.append(fix)                                                                                                                                                                                                                     
                            fix_index.append(idx)
                            fix_target_length.append(len(fix))
                            fix_input_length.append(input_length[idx])
                        elif model == 'ma-fix':
                            if not fix in buffer_batch_new[idx]:
                                buffer_batch_new[idx].append(fix)
                                buffer_batch_new_exp[idx].append(fix_infix)
                    except:
                        pass
                
                

            if model in ['reinforce', 'mapo']:
                try:
                    generate_infix = prefix_to_infix(generate_exp, target_length[idx])

                    if eval(generate_infix) == eval(num_ans[idx]):
                        if model == 'reinforce':
                            fix_target_list.append(exp.item()[:target_length[idx]])                                                                                                                                                                                                                     
                            fix_index.append(idx)
                            fix_target_length.append(len(exp))
                            fix_input_length.append(input_length[idx])
                        if model == 'mapo':
                            if not exp in buffer_batch_new[idx]:
                                buffer_batch_new[idx].append(exp.item()[:target_length[idx]])
                                buffer_batch_new_exp[idx].append(generate_infix)
                except:
                    pass

            if model in ['ma-fix', 'mapo']:
                for buffer_fix in buffer_batch_new[idx]:
                    fix_target_list.append(buffer_fix)                                                                                                                                                                                                                     
                    fix_index.append(idx)
                    fix_target_length.append(len(buffer_fix))
                    fix_input_length.append(input_length[idx])

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

    loss = torch.tensor([[0]])

    for j in range (num_iteration):
        if not j * mapo_batch_size + mapo_batch_size - 1 < len(fix_target_list):
            mapo_batch_size = len(fix_target_list) - ((j - 1) * mapo_batch_size + mapo_batch_size)
        target_length_mapo = fix_target_length[j * mapo_batch_size : (j * mapo_batch_size + mapo_batch_size)]
        idx_list = fix_index[j * mapo_batch_size : (j * mapo_batch_size + mapo_batch_size)]
        target_list = fix_target_list[j * mapo_batch_size : (j * mapo_batch_size + mapo_batch_size)]
        input_length_mapo = []
        num_pos_mapo = []
        num_size_mapo_batch = []    

        for k in range (mapo_batch_size):
            idx = idx_list[k]           
            input_length_mapo.append(input_length[idx])
            num_pos_mapo.append(num_pos[idx])
            num_size_mapo_batch.append(num_size_batch[idx])

        input_var_mapo = torch.zeros((max(input_length_mapo), mapo_batch_size),dtype=torch.long)
        target = torch.zeros((mapo_batch_size, max(target_length_mapo)), dtype = torch.long)

        for k in range (mapo_batch_size):
            idx = idx_list[k]
            input_var_mapo[:,k] = input_var[:,idx][:max(input_length_mapo)]

            target[k][:target_length_mapo[k]] = torch.LongTensor(target_list[k])

        seq_mask = []
        max_len = max(input_length_mapo)
        for i in input_length_mapo:
            seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
        seq_mask = torch.ByteTensor(seq_mask)

        num_mask = []
        max_num_size = max(num_size_mapo_batch) + 2
        for i in num_size_mapo_batch:
            d = i
            num_mask.append([0] * 2 + [0] * d + [1] * (max_num_size - d - 2))
        num_mask = torch.ByteTensor(num_mask)

        batch_size = len(input_length_mapo)

        target = torch.LongTensor(target).transpose(0, 1)

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

        for t in range(max_target_length):

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)

            target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, num_start)
            target[t] = target_t
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

        if USE_CUDA:
            # all_leafs = all_leafs.cuda()
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()

        # op_target = target < num_start
        # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)

        loss1 = masked_cross_entropy(all_node_outputs, target, target_length_mapo)
        #loss = loss1
        loss1.backward()
        loss += loss1
        # clip the grad
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
        # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

        # Update parameters with optimizers
        encoder_optimizer.step()
        predict_optimizer.step()
        generate_optimizer.step()
        merge_optimizer.step()

    loss = loss.item() if num_iteration == 0 else loss.item()/num_iteration
    return loss, buffer_batch_new, num_iteration, buffer_batch_new_exp  # , loss_0.item(), loss_1.item()
    # return 0


def evaluate_tree(input_batch, input_length, encoder, predict, generate, merge, output_lang, num_pos,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, 2+len(num_pos)).fill_(0)

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

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

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
                                            current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return [beams[0].out, beams[1].out, beams[2].out, beams[3].out, beams[4].out]


