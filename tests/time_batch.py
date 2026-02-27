from pathlib import Path
import time

import jsonargparse
import numpy as np
from tqdm import tqdm
import yaml


from crossgoose.data.dataset import FlowDataModule
from crossgoose.gridflow import GridFlow


def main(
    config_file:Path,
    n_tests:int
):
    
    
    with open(config_file, 'r') as f:
        full_cfg = yaml.safe_load(f)
    data_cfg = {'data': full_cfg.get('data', {})}

    parser = jsonargparse.ArgumentParser()

    parser.add_class_arguments(FlowDataModule, nested_key='data')
    cfg = parser.parse_string(yaml.dump(data_cfg))
    data = parser.instantiate_classes(cfg).data
    data.setup('fit')
    train_data = data.train_dataloader()

    times = []
    data_iter = iter(train_data)
    pbar = tqdm(range(n_tests))

    for _ in pbar:
        try:
            t0 = time.perf_counter()
            _ = next(data_iter)
            t1 = time.perf_counter()
        except StopIteration:
            data_iter = iter(train_data)
            t0 = time.perf_counter()
            _ = next(data_iter)
            t1 = time.perf_counter()
        times.append(t1-t0)
        
        times_s = np.array(times)
        pbar.set_description(
            f"timings: {np.mean(times_s):.2f}±{np.var(times_s):.2f}s"
        )

    print(f"batch loader: {np.mean(times_s):.2f}±{np.var(times_s):.2f}s")


if __name__ == "__main__":
    jsonargparse.auto_cli(main, as_positional=False, default_config_files=[
                          'tests/time_gridflow_query.yaml'])
