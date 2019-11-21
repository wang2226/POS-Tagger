# CIS 410/510: NLP - Assignment 2
# Due Nov 4th, 2019
# Haoran Wang (hwang8@cs.uoregon.edu)
# This is the implementation of a POS Tagger using HMM and Viterbi decoding

# To run:
# $ python ./POS_Tagger.py <training file> <file to tag> <output file> <ngram> <method to treat unknown words>
# <ngram>: bigram, trigram, exp
# <method to treat unknown words>: uniform, most_common_pos

import numpy as np
import sys


class POSTagger(object):

    def __init__(self):
        # transition count
        # count the times POS(t-1), POS(t), POS(t-2) appear together
        # { (POS(t-2),POS(t-1)): { POS(t) : count }}
        self.transition_count = {'START': {}}

        # emission
        # { POS : { word : count }}
        self.emission = {}

        # the POS and its count for each word
        # { word : { POS : count} ) }
        self.word_pos_dict = {}

        # the number of different POS tags in corpus
        self.PosSize = 0

        # POS-index correlation dictionary
        # {POS : index}
        self.label = {}

        # list of POS
        self.tag = []

        # Transition Matrices for trigram, bigram and unigram
        self.trans_matrix_trigram = []
        self.trans_matrix_bigram = []
        self.trans_matrix_unigram = []

        # Weighted Parameter
        self.lam3 = 0
        self.lam2 = 0
        self.lam1 = 0

        # the most common POS in the corpus
        self.most_common_pos = ''

        # bigram or trigram HMM
        self.ngram = ''

        # Methods to handle Emission probability for unknown words
        self.treat_unknown = ''

    def generate_hmm(self, path):
        # pointer for POS at t
        curr = ''
        # pointer for POS at t-1
        pre = 'START'
        # pointer for POS at t-2
        prepre = ''

        training = open(path, 'r')
        lines = training.readlines()

        # read training file
        for i in range(len(lines)):
            # get rid of \n
            lines[i] = lines[i].rstrip('\n').strip()
            # get rid of \t
            lines_field = lines[i].split('\t')

            # end of a sentence
            if len(lines_field) == 1:
                # add 'END' label
                pos = 'END'
                # because 'END' is new label to the dictionary, we need to handle its transition count
                # new (POS(t-2),POS(t-1)) entry, add to dictionary
                if (prepre, pre) not in self.transition_count:
                    self.transition_count[(prepre, pre)] = {pos: 1}
                # new POS for (POS(t-2), POS(t-1)), add to dictionary
                elif pos not in self.transition_count[(prepre, pre)]:
                    self.transition_count[(prepre, pre)][pos] = 1
                # already seen, increment
                else:
                    self.transition_count[(prepre, pre)][pos] += 1
                # reset pre pointer
                pre = 'START'
                continue

            # separate word and its POS
            word, pos = lines_field[0], lines_field[1]

            # For each word, record the POS it appears as and the count for the POS
            # new word, add the word and its POS to dictionary
            if word not in self.word_pos_dict:
                self.word_pos_dict[word] = {pos: 1}
            # new POS for existing word, add the POS and its count
            elif pos not in self.word_pos_dict[word]:
                self.word_pos_dict[word][pos] = 1
            # already seen, increment count
            else:
                self.word_pos_dict[word][pos] += 1

            # emission count
            # new POS, add the POS, its word to dictionary
            if pos not in self.emission:
                self.emission[pos] = {word: 1}
            # new word for existing POS, add the word and its count
            elif word not in self.emission[pos]:
                self.emission[pos][word] = 1
            # already seen, increment
            else:
                self.emission[pos][word] += 1

            # transition count
            # handle start probabilities, {'START' : {POS : count}}
            if pre == 'START':
                if pos not in self.transition_count['START']:
                    self.transition_count['START'][pos] = 1
                else:
                    self.transition_count['START'][pos] += 1
            # get transition count for [(POS(t-2), POS(t-1)), POS(t)]
            else:
                # new entry in the dictionary, add
                if (prepre, pre) not in self.transition_count:
                    self.transition_count[(prepre, pre)] = {pos: 1}
                # new POS(t) for (POS(t-2), POS(t-1)), add
                elif pos not in self.transition_count[(prepre, pre)]:
                    self.transition_count[(prepre, pre)][pos] = 1
                # already seen, increment
                else:
                    self.transition_count[(prepre, pre)][pos] += 1

            # update pointers
            prepre = pre
            pre = pos

        training.close()

        # print(self.word_pos_dict)
        # print(self.emission)
        # print(self.transition_count)

        # Get most common POS in the training corpus
        common_pos = {}
        for key in self.emission.keys():
            common_pos[key] = (sum(list(self.emission[key].values())))
            if common_pos[key] == max(common_pos.values()):
                self.most_common_pos = key
        # print(self.most_common_pos)

        # number of different POS
        self.PosSize = len(self.emission)

        # {POS : index}
        self.label = {Pos: enum for enum, Pos in enumerate(self.emission)}

        # rearrange
        tmp = [(self.label[Pos], Pos) for Pos in self.label]

        # Add index to 'START' and 'END'
        self.tag = [t[1] for t in tmp]
        self.tag.append('START')
        self.label.update({'END': self.PosSize, 'START': self.PosSize})

        # Transition probabilities
        # initialize a matrix for transition probabilities
        self.trans_matrix_trigram = np.zeros([self.PosSize + 1, self.PosSize + 1, self.PosSize + 1])

        for prepre_pre_combo in self.transition_count:
            if prepre_pre_combo == 'START':
                # index of 'START'
                index_prepre = self.label['START']
                index_pre = index_prepre
            else:
                # index of POS(t-2)
                index_prepre = self.label[prepre_pre_combo[0]]

                # index of POS(t-1)
                index_pre = self.label[prepre_pre_combo[1]]

            # iterate POS and add its entry to matrix
            # how many times trigram appears together
            # [index of POS(t-2), index of POS(t-1), index of POS(t)]
            for curr in self.transition_count[prepre_pre_combo]:
                self.trans_matrix_trigram[index_prepre, index_pre, self.label[curr]] \
                    = self.transition_count[prepre_pre_combo][curr]
        # print(self.trans_matrix_trigram)

        # smoothing, add one to transition count
        self.trans_matrix_trigram += 1

        # Trigram Transition Probabilities
        for mat in self.trans_matrix_trigram:
            #print(mat)
            for vec in mat:
                # calculate probabilities
                vec *= 1.0 / np.sum(vec)
        # print(self.trans_matrix_trigram)

        # sum down rows of TransMat
        # [total count POS(t-2), total count POS(t-1), total count POS(t)]
        self.trans_matrix_bigram = np.sum(self.trans_matrix_trigram, axis=0)
        # print(self.trans_matrix_bigram)
        for vec in self.trans_matrix_bigram:
            vec *= 1.0 / np.sum(vec)
        self.trans_matrix_bigram = self.trans_matrix_bigram * np.ones(
            [self.PosSize + 1, self.PosSize + 1, self.PosSize + 1])
        # print(self.trans_matrix_bigram)

        # for unigram transition probabilities
        self.trans_matrix_unigram = np.sum(self.trans_matrix_bigram, axis=0)
        for vec in self.trans_matrix_unigram:
            vec *= 1.0 / np.sum(vec)
        self.trans_matrix_unigram = self.trans_matrix_unigram * np.ones(
            [self.PosSize + 1, self.PosSize + 1, self.PosSize + 1])
        # print(self.trans_matrix_unigram)

        # print(self.emission_count)
        # emission probabilities
        for pos in self.emission:
            # vec = [word count of that pos]
            vec = list(self.emission[pos].values())
            total = {}
            for word in self.emission[pos]:
                total[word] = sum(vec)

            for word in self.emission[pos]:
                self.emission[pos][word] = self.emission[pos][word] / total[word]

        # Get lambda using Deleted-Interpolation
        tri_greater_bi = self.trans_matrix_trigram[1:, :-1] >= self.trans_matrix_bigram[1:, :-1]
        bi_greater_uni = self.trans_matrix_bigram[1:, :-1] >= self.trans_matrix_unigram[1:, :-1]

        self.lam3 = np.sum(self.trans_matrix_bigram[1:, :-1][tri_greater_bi])
        self.lam2 = np.sum(self.trans_matrix_bigram[1:, :-1][tri_greater_bi is not True])
        self.lam1 = np.sum(self.trans_matrix_unigram[1:, :-1][bi_greater_uni])

        if ngram == 'trigram':
            total = self.lam2 + self.lam3 + self.lam1
            self.lam3 = self.lam3 / total
            self.lam2 = self.lam3 / total
            self.lam1 = self.lam1 / total
        elif ngram == 'bigram':
            total = self.lam2 + self.lam1
            self.lam2 = self.lam2 / total
            self.lam1 = self.lam1 / total
        elif ngram == 'exp':
            total = self.lam3 + self.lam2
            self.lam3 = self.lam3 / total
            self.lam2 = self.lam2 / total

        # print(self.lam3)
        # print(self.lam2)
        # print(self.lam1)

    def emission_prob(self, word, treat_unknown):
        ret = []
        # known words
        if word in self.word_pos_dict:
            for pos in self.word_pos_dict[word].keys():
                ret.append((pos, self.emission[pos][word]))
        # unknown words, give a uniform probability (1/POS_Size for every POS)
        else:
            if treat_unknown == 'uniform':
                for pos in self.label:
                    ret.append((pos, 1.0 / tagger.PosSize))

            elif treat_unknown == 'most_common_pos':
                for pos in self.label:
                    if pos == self.most_common_pos:
                        ret.append((pos, 1.0))
                    else:
                        ret.append((pos, 0.0))

        return ret

    def viterbi(self, sequence, ngram, treat_unknown):
        # Let n be the length of the sequence
        n = len(sequence)

        # DP table
        # create a path probability matrix viterbi
        vtb = np.zeros([n + 2, self.PosSize + 1, self.PosSize + 1])

        # Back Pointer Table
        back_pointer = np.ones([n + 2, self.PosSize + 1, self.PosSize + 1]) * -1

        # Base Case
        vtb[0, self.label['START'], :] += 1

        # result
        ret = []
        for i in range(1, n + 1):
            word = sequence[i - 1]
            pos_emission = self.emission_prob(word, treat_unknown)
            for pos, emission in pos_emission:
                # calculate viterbi
                if ngram == 'trigram':
                    tmp = vtb[i - 1] * (self.lam3 * self.trans_matrix_trigram[:, :, self.label[pos]]
                                        + self.lam2 * self.trans_matrix_bigram[:, :, self.label[pos]]
                                        + self.lam1 * self.trans_matrix_unigram[:, :, self.label[pos]])
                elif ngram == 'bigram':
                    tmp = vtb[i - 1] * (self.lam2 * self.trans_matrix_bigram[:, :, self.label[pos]]
                                        + self.lam1 * self.trans_matrix_unigram[:, :, self.label[pos]])
                elif ngram == 'exp':
                    tmp = vtb[i - 1] * (self.lam3 * self.trans_matrix_trigram[:, :, self.label[pos]]
                                        + self.lam2 * self.trans_matrix_bigram[:, :, self.label[pos]])

                # get max viterbi
                vtb[i, :, self.label[pos]] = np.max(tmp, axis=0) * emission * 100
                # set back_pointer
                back_pointer[i, :, self.label[pos]] = np.argmax(tmp, axis=0)

        i = n + 1
        pos = 'END'

        if ngram == 'trigram':
            tmp = vtb[i - 1] * (self.lam3 * self.trans_matrix_trigram[:, :, self.label[pos]]
                                + self.lam2 * self.trans_matrix_bigram[:, :, self.label[pos]]
                                + self.lam1 * self.trans_matrix_unigram[:, :, self.label[pos]])
        elif ngram == 'bigram':
            tmp = vtb[i - 1] * (self.lam2 * self.trans_matrix_bigram[:, :, self.label[pos]]
                                + self.lam1 * self.trans_matrix_unigram[:, :, self.label[pos]])
        elif ngram == 'exp':
            tmp = vtb[i - 1] * (self.lam3 * self.trans_matrix_trigram[:, :, self.label[pos]]
                                        + self.lam2 * self.trans_matrix_bigram[:, :, self.label[pos]])

        vtb[i, :, self.label[pos]] = np.max(tmp, axis=0)
        back_pointer[i, :, self.label[pos]] = np.argmax(tmp, axis=0)

        to_pos = self.label['END']
        from_pos = int(np.argmax(vtb[i, :, to_pos]))
        pre_pos = int(back_pointer[i, from_pos, to_pos])

        for i in range(n, 0, -1):
            ret.append(self.tag[from_pos])
            to_pos = from_pos
            from_pos = pre_pos
            pre_pos = int(back_pointer[i, from_pos, to_pos])
        # print(ret)
        ret.reverse()
        return ret

    def pos_tag(self, file_to_tag, result_file, ngram, treat_unknown):
        tagging_file = open(file_to_tag, 'r')
        word = tagging_file.readline()

        sequence = []
        while word != '':
            word = word.strip('\n')
            if word != '':
                sequence.append(word)
            else:
                for (x, y) in zip(sequence, self.viterbi(sequence, ngram, treat_unknown)):
                    # print("%s\t%s\n" % (x, y))
                    result_file.write("%s\t%s\n" % (x, y))
                result_file.write("\n")
                sequence = []
            word = tagging_file.readline()
        tagging_file.close()


if __name__ == '__main__':

    if len(sys.argv) != 6:
        print("ERROR! INCORRECT NUMBER OF ARGUMENTS")

    training_file = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    ngram = sys.argv[4]
    treat_unknown = sys.argv[5]

    # instance of POSTagger class
    tagger = POSTagger()

    # generate HMM model from the training file
    tagger.generate_hmm(training_file)

    # tag file and write the result to a file
    result = open(output_file, "w")
    tagger.pos_tag(input_file, result, ngram, treat_unknown)
    result.close()
