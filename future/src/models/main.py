from evpi_pytorch import evpi
from model_helper import *


def main(args):
    post_ids_train = p.load(open(args.post_ids_train, 'rb'))
    post_ids_train = np.array(post_ids_train)
    post_vectors_train = p.load(open(args.post_vectors_train, 'rb'))
    ques_list_vectors_train = p.load(open(args.ques_list_vectors_train, 'rb'))
    ans_list_vectors_train = p.load(open(args.ans_list_vectors_train, 'rb'))

    labels = np.zeros((len(post_ids_train), 10), dtype=np.int32)
    labels[:, 0] = 1

    post_ids_test = p.load(open(args.post_ids_test, 'rb'))
    post_ids_test = np.array(post_ids_test)
    post_vectors_test = p.load(open(args.post_vectors_test, 'rb'))
    ques_list_vectors_test = p.load(open(args.ques_list_vectors_test, 'rb'))
    ans_list_vectors_test = p.load(open(args.ans_list_vectors_test, 'rb'))

    out_file = open(args.test_predictions_output, 'w')
    out_file.close()

    word_embeddings = p.load(open(args.word_embeddings, 'rb'))
    word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
    vocab_size = len(word_embeddings)
    word_emb_dim = len(word_embeddings[0])
    print 'word emb dim: ', word_emb_dim
    print 'vocab_size ', vocab_size, ', post_max_len ', args.post_max_len, ' ques_max_len ', args.ques_max_len, ' ans_max_len ', args.ans_max_len

    start = time.time()
    print 'generating data'
    train = generate_data(post_vectors_train, ques_list_vectors_train, ans_list_vectors_train, args)
    test = generate_data(post_vectors_test, ques_list_vectors_test, ans_list_vectors_test, args)
    train.append(post_ids_train)
    test.append(post_ids_test)
    print 'done! Time taken: ', time.time() - start

    print 'Size of training data: ', len(post_ids_train)
    print 'Size of test data: ', len(post_ids_test)

    if args.model == 'evpi':
        evpi(word_embeddings, vocab_size, word_emb_dim, args, train, test)
    else:
        raise ValueError('Uknown model {0}'.format(args.model))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--post_ids_train", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/ranking_clarification_questions/data/github_partial/post_ids_train.p')
    argparser.add_argument("--post_vectors_train", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/ranking_clarification_questions/data/github_partial/post_vectors_train.p')
    argparser.add_argument("--ques_list_vectors_train", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/ranking_clarification_questions/data/github_partial/ques_list_vectors_train.p')
    argparser.add_argument("--ans_list_vectors_train", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/ranking_clarification_questions/data/github_partial/ans_list_vectors_train.p')
    argparser.add_argument("--post_ids_test", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/ranking_clarification_questions/data/github_partial/post_ids_test.p')
    argparser.add_argument("--post_vectors_test", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/ranking_clarification_questions/data/github_partial/post_vectors_test.p')
    argparser.add_argument("--ques_list_vectors_test", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/ranking_clarification_questions/data/github_partial/ques_list_vectors_test.p')
    argparser.add_argument("--ans_list_vectors_test", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/ranking_clarification_questions/data/github_partial/ans_list_vectors_test.p')
    argparser.add_argument("--word_embeddings", type=str,
                           default='/Users/ciborowskaa/VCU/Research/BugReportQA/GAN_question_generation/embeddings/stackexchange_rao/word_embeddings.p')
    argparser.add_argument("--batch_size", type=int, default=256)
    argparser.add_argument("--no_of_epochs", type=int, default=20)
    argparser.add_argument("--hidden_dim", type=int, default=100)
    argparser.add_argument("--no_of_candidates", type=int, default=10)
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--rho", type=float, default=1e-5)
    argparser.add_argument("--post_max_len", type=int, default=300)
    argparser.add_argument("--ques_max_len", type=int, default=40)
    argparser.add_argument("--ans_max_len", type=int, default=40)
    argparser.add_argument("--test_predictions_output", type=str, default='predictions.txt')
    argparser.add_argument("--stdout_file", type=str, default='output.txt')
    argparser.add_argument("--model", type=str, default='evpi')
    args = argparser.parse_args()
    print args
    print ""
    main(args)
