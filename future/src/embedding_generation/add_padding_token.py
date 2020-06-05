import sys
import click


@click.command()
@click.option('--glove-vectors', required=True, help='File path to vectors.txt produced by Glove')
@click.option('--output-vectors', required=True, help='File path to save new vectors.txt')
def run(**args):
    glove_in = args['glove_vectors']
    glove_out = args['output_vectors']

    f_out = open(glove_out, 'w')
    with open(glove_in, 'r') as f_in:
        line = f_in.readline()
        emb_size = get_embeddings_size(line)
        padding_v = prepare_padding_vector(emb_size)

        # save padding vector
        f_out.write(padding_v + '\n')

        # copy rest as it was
        f_out.write(line)
        for line in f_in.readlines():
            f_out.write(line)
    f_out.close()


def get_embeddings_size(line):
    return len(line.split(' ')) - 1


def prepare_padding_vector(embedding_size, token='<PAD>'):
    return token + ' ' + ' '.join(['0.0' for i in range(0, embedding_size)])


if __name__ == '__main__':
    run(sys.argv[1:])
