from tensorboard import SummaryWriter

def create_writer(log_dir):
    return SummaryWriter(log_dir)

def log_to_tensorboard(writer, step, prefix, loss):
    """
    Log metrics to Tensorboard.
    """
    log_generic_to_tensorboard(writer, step, prefix, "loss", loss)

def log_generic_to_tensorboard(writer, step, prefix, metric, value):
    writer.add_scalar("{}/{}".format(prefix, metric), value)