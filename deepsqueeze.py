import argparse
import keras
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

EPS = sys.float_info.epsilon

def scale(df):
    range_ = df.agg(['min', 'max'])
    df = (df - range_.iloc[0]) / (range_.iloc[1] - range_.iloc[0] + EPS)
    return df, range_

def to_parquet(df, file, compression, level):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), file,
                   use_dictionary=True, compression=compression,
                   write_statistics=False, compression_level=level,
                   data_page_version='2.0', store_schema=False)
    return os.path.getsize(file)

def compress(file, batch, compression, delimiter, error, epochs, level, loss,
             output, **kwargs):
    out = f'.{file.split("/")[-1] if output is None else output}'
    os.system(f'rm -rf {out} && mkdir {out}')
    pd.DataFrame([error]).to_csv(f'{out}/error.csv', index=False)

    X, range_ = scale(pd.read_csv(file, sep=delimiter))
    range_.to_csv(f'{out}/range.csv', index=False)

    encoder = keras.Sequential([
                  keras.Input((X.shape[1],)),
                  keras.layers.Dense(X.shape[1] * 2, activation='leaky_relu'),
                  keras.layers.Dense(X.shape[1] * 2, activation='leaky_relu'),
                  keras.layers.Dense(1, activation='sigmoid')
              ], name='encoder')
    decoder = keras.Sequential([
                  keras.Input((1,)),
                  keras.layers.Dense(X.shape[1] * 2, activation='leaky_relu'),
                  keras.layers.Dense(X.shape[1] * 2, activation='leaky_relu'),
                  keras.layers.Dense(X.shape[1], activation='sigmoid')
              ], name='decoder')
    autoencoder = keras.Sequential([encoder, decoder])
    autoencoder.compile('adam', loss)
    autoencoder.fit(X, X, batch_size=batch, epochs=epochs)
    decoder.save(f'{out}/decoder.keras')

    prev = sys.maxsize
    for i in range(10):
        Z = encoder.predict(X).round(i)
        codes = pd.DataFrame(Z * 10.0 ** i).astype(int)
        size = to_parquet(codes, f'{out}/codes-{i}.pq', compression, level)

        X_ = pd.DataFrame(decoder.predict(Z), columns=X.columns)
        deltas = ((X - X_) / error).astype(int)
        size += to_parquet(deltas, f'{out}/deltas-{i}.pq', compression, level)

        if size > prev:
            pd.DataFrame([i - 1]).to_csv(f'{out}/quantize.csv', index=False)
            os.system(f'rm {out}/*-{i}.pq')
            break

        prev = size
        if i != 0:
            os.system(f'rm {out}/*-{i - 1}.pq')

    os.system(f'tar czf {out[1:]}.tar.gz {out} && rm -rf {out}')

def decompress(file, check, delimiter, output, **kwargs):
    out = f'.{file.split("/")[-1].split(".tar.gz")[0]}'
    os.system(f'tar xzf {file}')

    quantize = pd.read_csv(f'{out}/quantize.csv').iloc[0, 0]
    codes = pd.read_parquet(f'{out}/codes-{quantize}.pq')
    Z = codes / 10.0 ** quantize

    decoder = keras.saving.load_model(f'{out}/decoder.keras')
    X_ = decoder.predict(Z)

    error = pd.read_csv(f'{out}/error.csv').iloc[0, 0]
    deltas = pd.read_parquet(f'{out}/deltas-{quantize}.pq')
    X_ = X_ + deltas * error

    if check is not None:
        X, _ = scale(pd.read_csv(check, sep=delimiter))
        diffs = (X - X_).abs() >= error
        if diffs.any().any():
            print(f'Columns exceeding {error} error:\n{diffs.any()}')

    range_ = pd.read_csv(f'{out}/range.csv')
    X_ = X_ * (range_.iloc[1] - range_.iloc[0] + EPS) + range_.iloc[0]
    X_.to_csv(out[1:] if output is None else output, index=False)
    os.system(f'rm -rf {out}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-b', '--batch', default=64, type=int)
    parser.add_argument('-c', '--compression', default='gzip')
    parser.add_argument('-C', '--check')
    parser.add_argument('-d', '--decompress', action='store_true')
    parser.add_argument('-D', '--delimiter', default=',')
    parser.add_argument('-e', '--error', default=1e-3, type=float)
    parser.add_argument('-E', '--epochs', default=10, type=int)
    parser.add_argument('-l', '--level', default=6, type=int)
    parser.add_argument('-L', '--loss', default='mae')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    if args.decompress:
        decompress(**vars(args))
    else:
        compress(**vars(args))

if __name__ == '__main__':
    main()
