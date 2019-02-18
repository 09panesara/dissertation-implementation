'''
Implementation of Baum Welch learning algorithm for parameter estimation for viterbi
'''
import numpy as np

class BaumWelch:
    def __init__(self, emission_mtx, transition_mtx, initial_probs):
        self.emission_mtx = emission_mtx
        self.transition_mtx = transition_mtx
        self.no_hidden_states = len(transition_mtx) # includes start and end artificial states
        self.no_emission_symbols = len(emission_mtx)
        self.initial_probs = initial_probs
        # calculate observed sequence mapping

    def set_sequence(self, observed, observed_mapping):
        self.observed = observed
        self.x = [observed_mapping[x_t] for x_t in self.observed]
        self.len_sequence = len(observed)


    def forward_algorithm(self):
        transition_mtx = self.transition_mtx
        emission_mtx = self.emission_mtx
        x = self.x
        initial_probs = self.initial_probs

        # x = symbols of string in form of indices into emission_mtx
        T = len(x)

        n = len(transition_mtx[0]) # emission matrix contains entries for start and end state
        no_emission_symbols = len(zip(*emission_mtx)[0])

        # create n x (k+1) matrix for n hidden states, k symbols in observed sequence + 1 for start
        f = np.zeros((n,T)) # check if k+1 needed
        # start state
        f[:, 0] = initial_probs
        # intermediate states
        for t in range(1, n):
            for i in range(1, T):
                f_i = 0
                if t == 1:
                    f_i += (emission_mtx[i, x[0]]*initial_probs[i])
                else:
                    for j in range(1, no_emission_symbols):
                        f_i += (emission_mtx[j,i] * f[j,t-1])
                    f_i *= emission_mtx[i,x[t]]
                f[j,t] = f_i
        # end state
        return f



    def backward_algorithm(self):
        transition_mtx = self.transition_mtx
        emission_mtx = self.emission_mtx
        x = self.x
        initial_probs = self.initial_probs

        # x = symbols of string in form of indices into emission_mtx
        T = len(x)

        n = len(transition_mtx[0])  # emission matrix contains entries for start and end state
        no_emission_symbols = len(zip(*emission_mtx)[0])

        b = np.zeros((n, T))
        b[:, T] = np.ones(n) # rightmost column filled with 1's since initial state is assumed given

        for t in range(1, n):
            for i in range(T-1, 1):
                b_i = 0
                for j in range(1, no_emission_symbols):
                    b_i += (emission_mtx[j, i] * b[j, t + 1])
                b_i *= emission_mtx[i, x[t]]
                b[j, t] = b_i
                # https: // www.slideshare.net / ananth / an - overview - of - hidden - markov - models - HMM



        return b


    def baum_welch(self, sequences, no_iters):
        i = 0
        n = 0

        A_mat = self.transition_mtx
        O_mat = self.emission_mtx

        while (n < no_iters): # iterate for max number iterations or until have reached convergence
            old_A_mat = A_mat
            old_O_mat = O_mat
            A = np.zeros((self.no_hidden_states, self.no_hidden_states))
            O = np.zeros((self.no_hidden_states, self.no_emission_symbols))

            for seq in sequences:
                self.set_sequence(seq)
                f = self.forward_algorithm()
                b = self.backward_algorithm()
                P = f * b  # probability matrix
                P = P / np.sum(P, 0)
                x = self.x

                len_sequence = self.len_sequence

                # Update A
                for k in range(self.no_hidden_states):
                    for l in range(self.no_hidden_states):
                        for i in range(len_sequence):
                            A[k, l] += (f[k, i]*A_mat[k,l]*O_mat[l, self.x[i+1]] * b[l, i+1])/P

                # Update E
                for k in range(self.no_hidden_states):
                    for o in range(self.no_emission_symbols):
                        O[k, o] = O[k,o] + (f[k, i]*b[l, i])/P if self.x[i] == o else O[k,o]

            # Update transition matrix, emission matrix after updated A, E for each sequence
            for k in range(self.no_hidden_states):
                for l in range(self.no_hidden_states):
                    A_mat[k, l] = A[k, l]/np.sum(A[k, :])

            for k in range(self.no_hidden_states):
                for o in range(self.no_emission_symbols):
                    O_mat[k, o] = O[k, o]/np.sum(O[k, :])

            # Compute convergence
            if np.linalg.norm(old_A_mat - A_mat) < .00001 and np.linalg.norm(old_O_mat - O_mat) < .00001:
                break

        self.transition_mtx = A_mat
        self.emission_mtx = O_mat






# http://ab.inf.uni-tuebingen.de/teaching/ss04/abi2/AlBiII-SS2004-Huson.pdf
    '''
    see implementation here: https://gist.github.com/dougalsutherland/1329976
    '''