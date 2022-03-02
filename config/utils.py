import configparser


def read_config(args):
    config = configparser.ConfigParser()
    # config_path = args.config
    config_path = "example_configs\config_kitti_0006_example_train.txt"

    config.read(config_path)
    sections = config.sections()
    print(sections)
    for sec in sections:
        ops = config.options(sec)
        for op in ops:
            value = config.get(sec, op)
            try:
                res = eval(value)
                if isinstance(res, float):
                    setattr(args, op, res)
                elif isinstance(res, int):
                    setattr(args, op, res)
                else:
                    setattr(args, op, value)
            except Exception:
                setattr(args, op, value)
            
    return args


if __name__ == '__main__':
    args = {}
    args = read_config(args)
    print(args)
