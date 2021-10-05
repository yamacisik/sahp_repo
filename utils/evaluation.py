import numpy as np
import tqdm
from models.base import SeqGenerator
from utils.util import get_batch
from train_functions.train_sahp import MaskBatch
import math
def generate_multiple_sequences(generator: SeqGenerator, tmax: float, n_gen_seq: int = 100):
    """

    Args:
        generator:
        tmax: end time for the simulations
        n_gen_seq: number of samples to take
    """
    print("tmax:", tmax)
    # Build a statistic for the no. of events
    gen_seq_lengths = []
    gen_seq_types_lengths = []
    for i in range(n_gen_seq):
        print('Generating the {} sequence'.format(i))
        generator.generate_sequence(tmax, record_intensity=False)
        gen_seq_times = generator.event_times
        gen_seq_types = np.array(generator.event_types)
        gen_seq_lengths.append(len(gen_seq_times))
        gen_seq_types_lengths.append([
            (gen_seq_types == i).sum() for i in range(generator.model.input_size)
        ])
    gen_seq_lengths = np.array(gen_seq_lengths)
    gen_seq_types_lengths = np.array(gen_seq_types_lengths)

    print("Mean generated sequence length: {}".format(gen_seq_lengths.mean()))
    print("Generated sequence length std. dev: {}".format(gen_seq_lengths.std()))
    return gen_seq_lengths, gen_seq_types_lengths


def predict_test(model, seq_times, seq_types, seq_lengths, pad, device='cpu',
                 hmax: float = 40., use_jupyter: bool = False, rnn: bool = True):
    """Run predictions on testing dataset

    Args:
        seq_lengths:
        seq_types:
        seq_times:
        model:
        hmax:
        use_jupyter:

    Returns:

    """
    incr_estimates = []
    incr_real = []
    incr_errors = []
    types_real = []
    types_estimates = []
    test_size = seq_times.shape[0]
    if use_jupyter:
        index_range_ = tqdm.tnrange(test_size)
    else:
        index_range_ = tqdm.trange(test_size)
    for index_ in index_range_:
        _seq_data = (seq_times[index_],
                     seq_types[index_],
                     seq_lengths[index_])
        if rnn:
            est, real_dt, err, real_type, est_type = model.read_predict(*_seq_data, hmax)
        else:
            est, real_dt, err, real_type, est_type = model.read_predict(*_seq_data, pad, device, hmax)

        if err != err: # is nan
            continue
        incr_estimates.append(est)
        incr_real.append(real_dt)
        incr_errors.append(err)
        types_real.append(real_type)
        types_estimates.append(est_type)

    incr_real = np.asarray(incr_real)
    incr_estimates = np.asarray(incr_estimates)
    types_real = np.asarray(types_real)
    types_estimates = np.asarray(types_estimates)

    return incr_estimates,incr_real, types_real, types_estimates


def get_intensities_from_sahp(model, test_data, batch_size=32):
    device = model.device

    test_seq_times, test_seq_types, test_seq_lengths, test_seq_intensities = test_data

    test_seq_lengths, reorder_indices_test = test_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    test_seq_times = test_seq_times[reorder_indices_test]
    test_seq_types = test_seq_types[reorder_indices_test]
    test_seq_intensities = test_seq_intensities[reorder_indices_test]

    test_size = test_seq_times.size(0)
    test_loop_range = list(range(0, test_size, batch_size))

    model.eval()

    all_intensities = []
    all_predicted_intensities = []

    for i_batch in test_loop_range:

        batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
            get_batch(batch_size, i_batch, model, test_seq_lengths, test_seq_times, test_seq_types, rnn=False)
        batch_seq_types = batch_seq_types[:, 1:]
        batch_intensities = test_seq_intensities[i_batch:i_batch + batch_size][:, 1:]

        masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim,
                                     device=device)  # exclude the first added event
        model.forward(batch_dt, masked_seq_types.src, masked_seq_types.src_mask)
        # nll = model.compute_loss(batch_seq_times, batch_onehot)

        type_embedding = model.type_emb(masked_seq_types.src) * math.sqrt(model.d_model)  #
        position_embedding = model.position_emb(masked_seq_types.src, batch_dt)

        x = type_embedding + position_embedding

        dt_seq = batch_seq_times[:, 1:] - batch_seq_times[:, :-1]
        cell_t = model.state_decay(model.converge_point, model.start_point, model.omega, dt_seq[:, :, None])

        n_batch = test_seq_times.size(0)
        n_times = test_seq_times.size(1) - 1
        device = dt_seq.device
        # Get the intensity process
        intens_at_evs = model.intensity_layer(cell_t)
        intens_at_evs = intens_at_evs.squeeze(-1)

        index = 0
        for i in batch_seq_lengths:
            predicted_intensities = intens_at_evs[index, :i]
            actual_intensities = batch_intensities[index, :i]
            all_intensities.append(actual_intensities.cpu().detach().numpy())
            all_predicted_intensities.append(predicted_intensities.cpu().detach().numpy())

            index += 1

    all_predicted_intensities = np.concatenate(all_predicted_intensities)
    all_intensities = np.concatenate(all_intensities)

    return all_predicted_intensities, all_intensities

