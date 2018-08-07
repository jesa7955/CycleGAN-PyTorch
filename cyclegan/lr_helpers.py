def get_lambda_rule(opts):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch + opts.start_epoch - opts.decay_epoch) / float(opts.epochs - opts.decay_epoch)
    return lambda_rule
