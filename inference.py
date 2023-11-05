#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu)
# updated 10/05/2021 by Gary Lee (Fall 2021)
#
# Modified by: <your name here!>


# use this to enable/disable graphics
enable_graphics = True

import sys
import numpy as np
import robot

if enable_graphics:
    import graphics


# -----------------------------------------------------------------------------
# Functions for you to implement
#
def log_prob(prob):
    """compute log probability"""
    with np.errstate(divide="ignore"):
        log_prob = np.log(prob)
    log_prob[
        prob <= 0
    ] = -np.inf  # Set log probability to -inf where prob is 0 or negative
    return log_prob


def create_state_index_mapping(all_possible_hidden_states):
    """Creates a mapping from hidden states to indices and vice versa."""
    state_to_index = {state: i for i, state in enumerate(all_possible_hidden_states)}
    index_to_state = {i: state for state, i in state_to_index.items()}
    return state_to_index, index_to_state


def distribution_to_vector(distribution, state_to_index):
    """Converts a distribution dictionary to a vector using the state index mapping."""
    vector = np.zeros(len(state_to_index))
    for state, prob in distribution.items():
        vector[state_to_index[state]] = prob
    return vector


def matrix_from_transition_model(
    transition_model, all_possible_hidden_states, state_to_index
):
    """Creates a transition matrix from the transition model using the state index mapping."""
    size = len(all_possible_hidden_states)
    matrix = np.zeros((size, size))
    for from_state in all_possible_hidden_states:
        from_index = state_to_index[from_state]
        for to_state, prob in transition_model(from_state).items():
            to_index = state_to_index[to_state]
            matrix[from_index, to_index] = prob
    return matrix


def matrix_from_observation_model(
    observation_model,
    all_possible_hidden_states,
    all_possible_observed_states,
    state_to_index,
):
    """Creates an observation matrix from the observation model for a specific observation."""
    num_states = len(all_possible_hidden_states)
    num_observations = len(all_possible_observed_states)
    matrix = np.zeros((num_states, num_observations))
    for state in all_possible_hidden_states:
        state_index = state_to_index[state]
        for observation, prob in observation_model(state).items():
            if observation is not None:  # handle None observation separately if needed
                observation_index = all_possible_observed_states.index(observation)
                matrix[state_index, observation_index] = prob
    return matrix

def forward_backward(
    all_possible_hidden_states,
    all_possible_observed_states,
    prior_distribution,
    transition_model,
    observation_model,
    observations,
):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    all_possible_observed_states: a list of possible observed states

    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    # precompute transtition matrix and observation matrix
    state_to_index, index_to_state = create_state_index_mapping(
        all_possible_hidden_states
    )
    log_transition_matrix = log_prob(
        matrix_from_transition_model(
            transition_model, all_possible_hidden_states, state_to_index
        )
    )
    log_observation_matrix = log_prob(
        matrix_from_observation_model(
            observation_model,
            all_possible_hidden_states,
            all_possible_observed_states,
            state_to_index,
        )
    )  # a |X|*|Y| matrix
    prior_vector = distribution_to_vector(prior_distribution, state_to_index)
    # forward_messages = [None] * num_time_steps
    forward_messages = np.zeros((num_time_steps, len(all_possible_hidden_states)))
    forward_messages[0] = log_prob(prior_vector)

    # TODO: Compute the forward messages
    for t in range(1, num_time_steps):
        if observations[t - 1] is None:
            log_observation_vector = log_prob(np.zeros(len(all_possible_hidden_states)))

        else:
            observation_index = all_possible_observed_states.index(observations[t - 1])
            log_observation_vector = log_observation_matrix[:, observation_index]

        # avoid numerical issues with the exp and log
        log_forward_message = np.logaddexp.reduce(
            forward_messages[t - 1][:, None]
            + log_transition_matrix
            + log_observation_vector,
            axis=1,
        )
        forward_messages[t, :] = log_forward_message.squeeze()

    # backward_messages = [None] * num_time_steps
    backward_messages = np.zeros((num_time_steps, len(all_possible_hidden_states)))
    backward_messages[-1] = np.zeros(len(all_possible_hidden_states))
    # TODO: Compute the backward messages
    for t in range(num_time_steps - 2, -1, -1):
        if observations[t + 1] is not None:
            observation_index = all_possible_observed_states.index(observations[t + 1])
            log_observation_vector = log_observation_matrix[:, observation_index]
        else:
            log_observation_vector = log_prob(np.zeros(len(all_possible_hidden_states)))

        log_backward_message = np.logaddexp.reduce(
            backward_messages[t + 1][:,None]
            + log_transition_matrix.T
            + log_observation_vector,
            axis=1,
        )
        backward_messages[t, :] = log_backward_message.squeeze()
        print(backward_messages[t,:])

    marginals = [None] * num_time_steps  # remove this
    # TODO: Compute the marginals
    for t in range(num_time_steps):
        if observations[t] is not None:
            observation_index = all_possible_observed_states.index(observations[t])
            log_observation_vector = log_observation_matrix[:, observation_index]
        else:
            log_observation_vector = np.zeros(len(all_possible_hidden_states))

        log_marginal = forward_messages[t] + backward_messages[t] + log_observation_vector
        
        # Normalize to prevent numerical instability
        log_max = np.max(log_marginal)
        if np.isinf(log_max):  # if log_max is -inf, then all probabilities are zero
            marginal = np.zeros(len(all_possible_hidden_states))
        else:
            marginal = np.exp(log_marginal)
            
        # Convert the marginal distribution back to the Distribution form
        marginals[t] = robot.Distribution(
            {index_to_state[i]: prob for i, prob in enumerate(marginal)}
        )
        marginals[t].renormalize()

    return marginals


def Viterbi(
    all_possible_hidden_states,
    all_possible_observed_states,
    prior_distribution,
    transition_model,
    observation_model,
    observations,
):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps  # remove this

    return estimated_hidden_states


def second_best(
    all_possible_hidden_states,
    all_possible_observed_states,
    prior_distribution,
    transition_model,
    observation_model,
    observations,
):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: This is for you to implement

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps  # remove this

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#


def generate_data(
    initial_distribution,
    transition_model,
    observation_model,
    num_time_steps,
    make_some_observations_missing=False,
    random_seed=None,
):
    # generate samples from a hidden Markov model given an initial
    # distribution, transition model, observation model, and number of time
    # steps, generate samples from the corresponding hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = initial_distribution().sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < 0.1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

if __name__ == "__main__":
    # flags
    make_some_observations_missing = False
    use_graphics = enable_graphics
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == "--missing":
            make_some_observations_missing = True
        elif arg == "--nographics":
            use_graphics = False
        elif arg.startswith("--load="):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = generate_data(
            robot.initial_distribution,
            robot.transition_model,
            robot.observation_model,
            num_time_steps,
            make_some_observations_missing,
        )

    all_possible_hidden_states = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution = robot.initial_distribution()

    print("Running forward-backward...")
    marginals = forward_backward(
        all_possible_hidden_states,
        all_possible_observed_states,
        prior_distribution,
        robot.transition_model,
        robot.observation_model,
        observations,
    )
    print("\n")

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        top_10_states = sorted(
            marginals[timestep].items(), key=lambda x: x[-1], reverse=True
        )[:10]
        print([s for s in top_10_states if s[-1] > 0])
    else:
        print("*No marginal computed*")
    print("\n")

    timestep = 0
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        top_10_states = sorted(
            marginals[timestep].items(), key=lambda x: x[-1], reverse=True
        )[:10]
        print([s for s in top_10_states if s[-1] > 0])
    else:
        print("*No marginal computed*")
    print("\n")

    print("Running Viterbi...")
    estimated_states = Viterbi(
        all_possible_hidden_states,
        all_possible_observed_states,
        prior_distribution,
        robot.transition_model,
        robot.observation_model,
        observations,
    )
    print("\n")

    if num_time_steps > 10:
        print("Last 10 hidden states in the MAP estimate:")
        for time_step in range(num_time_steps - 10, num_time_steps):
            if estimated_states[time_step] is None:
                print("Missing")
            else:
                print(estimated_states[time_step])
        print("\n")

        print("Finding second-best MAP estimate...")
        estimated_states2 = second_best(
            all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            robot.transition_model,
            robot.observation_model,
            observations,
        )
        print("\n")

        print("Last 10 hidden states in the second-best MAP estimate:")
        for time_step in range(num_time_steps - 10 - 1, num_time_steps):
            if estimated_states2[time_step] is None:
                print("Missing")
            else:
                print(estimated_states2[time_step])
        print("\n")

        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states[time_step] != hidden_states[time_step]:
                difference += 1
        print(
            "Number of differences between MAP estimate and true hidden " + "states:",
            difference,
        )
        true_prob = robot.sequence_prob(
            hidden_states,
            robot.transition_model,
            robot.observation_model,
            prior_distribution,
            observations,
        )
        print("True Sequence Prob:", true_prob)
        map_prob = robot.sequence_prob(
            estimated_states,
            robot.transition_model,
            robot.observation_model,
            prior_distribution,
            observations,
        )
        print("MAP Estimate Prob:", map_prob)

        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states2[time_step] != hidden_states[time_step]:
                difference += 1
        print(
            "Number of differences between second-best MAP estimate and "
            + "true hidden states:",
            difference,
        )
        map_prob2 = robot.sequence_prob(
            estimated_states2,
            robot.transition_model,
            robot.observation_model,
            prior_distribution,
            observations,
        )
        print("Second-best MAP Estimate Prob:", map_prob2)

        difference = 0
        for time_step in range(num_time_steps):
            if estimated_states[time_step] != estimated_states2[time_step]:
                difference += 1
        print(
            "Number of differences between MAP and second-best MAP " + "estimates:",
            difference,
        )

    # display
    if use_graphics:
        app = graphics.playback_positions(
            hidden_states, observations, estimated_states, marginals
        )
        app.mainloop()
