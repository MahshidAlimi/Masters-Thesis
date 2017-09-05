import numpy as np

def forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]

    alpha = np.zeros((N, S))

    # base case
    alpha[0, :] = pi * O[:, observations[0]]

    # recursive case
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                alpha[i, s2] += alpha[i - 1, s1] * A[s1, s2] * O[s2, observations[i]]

    return (alpha, np.sum(alpha[N - 1, :]))


def Prep_Forward(obs):
    #Create transition,emission and initial matrix.
    count_1 = sum(obs)
    count_0 = len(obs) - count_1
    init_probs = [count_0 / len(obs), count_1 / len(obs)]

    same = [0] * 2
    diff = [0] * 2
    for i in range(1, len(obs)):
        if obs[i - 1] == obs[i]:
            same[obs[i]] += 1
        else:
            if obs[i - 1] == 0:
                diff[0] += 1
            else:
                diff[1] += 1
    len_0 = same[0] + diff[0]
    len_1 = same[1] + diff[1]
    try:
        transition_0 = [same[0] / len_0, diff[0] / len_0]
    except:
        transition_0 = [0., 0.]
    try:
        transition_1 = [diff[1] / len_1, same[1] / len_1]
    except:
        transition_1 = [0., 0.]
    trans = np.vstack((transition_0, transition_1))

    em_0 = [transition_1[0], transition_1[1]]
    em_1 = [transition_0[1], transition_0[0]]
    emission = np.vstack((em_0, em_1))
    return np.array(init_probs), trans, emission