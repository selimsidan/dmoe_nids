# Data

The experiments use four NetFlow v3 (NF-v3) datasets. They are not
redistributed with this repository: download them from the publisher.

## Sources and citations

| Dataset (registry key) | Shown in tables as | Official page | Built from |
|---|---|---|---|
| `NF-UNSW-NB15-v3` | NF-UNSW-NB15-v3 | https://espace.library.uq.edu.au/view/UQ:6e0eda1 | UNSW-NB15 |
| `NF-ToN-IoT-v3` | NF-ToN-IoT-v3 | https://espace.library.uq.edu.au/view/UQ:44d7c5e | ToN-IoT |
| `NF-BoT-IoT-v3` | NF-BoT-IoT-v3 | https://espace.library.uq.edu.au/view/UQ:73c4ebc | BoT-IoT |
| `NF-CICIDS2018-v3` | NF-CSE-CIC-IDS2018-v3 | https://espace.library.uq.edu.au/view/UQ:ece9b83 | CSE-CIC-IDS2018 |

The pages are the University of Queensland (UQ eSpace) records named in each
dataset's download package. Licensing details are on those pages.

* NF-v3 release: arXiv:2503.04404. TODO: full reference.
* Original datasets: TODO: references for UNSW-NB15, ToN-IoT, BoT-IoT and CSE-CIC-IDS2018.
* Licences: TODO: summarize the licence of each dataset from its page.

The same information is machine-readable in `configs/sources.yaml`.

## Folder layout

All four CSVs live under one folder, `DATA_ROOT`:

```
DATA_ROOT/
  NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.csv
  NF-ToN-IoT-v3/data/NF-ToN-IoT-v3.csv
  NF-BoT-IoT-v3/data/NF-BoT-IoT-v3.csv
  NF-CICIDS2018-v3/data/NF-CICIDS2018-v3.csv
```

Each download is a package (a BagIt folder) whose `data/` subfolder already
holds the CSV, so unpacking it into `DATA_ROOT/<dataset>/` gives exactly this
layout. Other files in the package (`bag-info.txt`, `manifest-sha1.txt`,
`data/NetFlow_v3_Features.csv`) are ignored.

`DATA_ROOT` is chosen by `--data-root`, the environment variable
`NIDS_DATA_DIR`, or defaults to `./data`. In the Colab notebooks it defaults to
`MyDrive/NIDS_datasets` (or the Colab secret `NIDS_DATA_DIR`).

## Step by step (Google Drive, for the Colab notebooks)

1. On each official page above, download the dataset package.
2. Unpack it. You should see a folder containing `data/<dataset>.csv`.
3. In Google Drive create `MyDrive/NIDS_datasets/` and inside it one folder per
   dataset, named exactly like the registry key (`NF-UNSW-NB15-v3`, ...).
4. Upload each dataset's `data/` folder into its dataset folder.
5. Open `notebooks/01_train_method.ipynb` in Colab, keep the default
   SETTINGS, and run it: its preflight cell checks the files before training.

Tip: the CSVs are large (tens of GB together). If you can obtain a direct
download link, downloading inside a Colab session straight into the mounted
Drive folder is much faster than uploading from a laptop. When a link is added
to `configs/sources.yaml` (`direct_url`), `python scripts/download_data.py`
does this automatically.

## Checking the files

```bash
python scripts/verify_data.py              # existence, columns, row counts, SHA-1
python scripts/verify_data.py --quick      # existence and columns only
```

Expected values are in `configs/checksums.json`. The SHA-1 values come from
the publisher's `manifest-sha1.txt`. Row counts (data rows, header excluded):

| Dataset | Rows |
|---|---|
| NF-UNSW-NB15-v3 | 2,365,424 |
| NF-ToN-IoT-v3 | 27,520,260 |
| NF-BoT-IoT-v3 | 16,993,808 |
| NF-CICIDS2018-v3 | 20,115,529 |

TODO: confirm these counts and record the SHA-256 and sizes by running
`python scripts/compute_checksums.py --write` once on the full files.

Hashing a large CSV takes minutes; the result is cached in
`OUTPUT_ROOT/cache/source_digests.json`, so it happens once per file.

## Columns

Each CSV has 53 NetFlow features plus two label columns.

**Never read** (identity, time and the binary label):
`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`,
`FLOW_START_MILLISECONDS`, `FLOW_END_MILLISECONDS`, `Label`.

**Label:** `Attack` (the attack name; `Benign` for normal traffic).

**The 47 model inputs, in model order:**

| # | Column | Description (from the publisher's feature list) |
|---|---|---|
| 1 | PROTOCOL | IP protocol identifier byte |
| 2 | L7_PROTO | Layer 7 protocol (numeric) |
| 3 | IN_BYTES | Incoming number of bytes |
| 4 | OUT_BYTES | Outgoing number of bytes |
| 5 | IN_PKTS | Incoming number of packets |
| 6 | OUT_PKTS | Outgoing number of packets |
| 7 | TCP_FLAGS | Cumulative of all TCP flags |
| 8 | CLIENT_TCP_FLAGS | Cumulative of all client TCP flags |
| 9 | SERVER_TCP_FLAGS | Cumulative of all server TCP flags |
| 10 | FLOW_DURATION_MILLISECONDS | Flow duration in milliseconds |
| 11 | DURATION_IN | Client to server stream duration (ms) |
| 12 | DURATION_OUT | Server to client stream duration (ms) |
| 13 | MIN_TTL | Minimum flow TTL |
| 14 | MAX_TTL | Maximum flow TTL |
| 15 | LONGEST_FLOW_PKT | Longest packet (bytes) of the flow |
| 16 | SHORTEST_FLOW_PKT | Shortest packet (bytes) of the flow |
| 17 | MIN_IP_PKT_LEN | Length of the smallest IP packet observed |
| 18 | MAX_IP_PKT_LEN | Length of the largest IP packet observed |
| 19 | SRC_TO_DST_SECOND_BYTES | Source to destination bytes per second |
| 20 | DST_TO_SRC_SECOND_BYTES | Destination to source bytes per second |
| 21 | RETRANSMITTED_IN_BYTES | Retransmitted TCP bytes (src to dst) |
| 22 | RETRANSMITTED_IN_PKTS | Retransmitted TCP packets (src to dst) |
| 23 | RETRANSMITTED_OUT_BYTES | Retransmitted TCP bytes (dst to src) |
| 24 | RETRANSMITTED_OUT_PKTS | Retransmitted TCP packets (dst to src) |
| 25 | SRC_TO_DST_AVG_THROUGHPUT | Source to destination average throughput (bps) |
| 26 | DST_TO_SRC_AVG_THROUGHPUT | Destination to source average throughput (bps) |
| 27 | NUM_PKTS_UP_TO_128_BYTES | Packets with IP size up to 128 |
| 28 | NUM_PKTS_128_TO_256_BYTES | Packets with IP size in (128, 256] |
| 29 | NUM_PKTS_256_TO_512_BYTES | Packets with IP size in (256, 512] |
| 30 | NUM_PKTS_512_TO_1024_BYTES | Packets with IP size in (512, 1024] |
| 31 | NUM_PKTS_1024_TO_1514_BYTES | Packets with IP size in (1024, 1514] |
| 32 | TCP_WIN_MAX_IN | Maximum TCP window (src to dst) |
| 33 | TCP_WIN_MAX_OUT | Maximum TCP window (dst to src) |
| 34 | ICMP_TYPE | ICMP type times 256 plus ICMP code |
| 35 | ICMP_IPV4_TYPE | ICMP type |
| 36 | DNS_QUERY_ID | DNS query transaction id |
| 37 | DNS_QUERY_TYPE | DNS query type |
| 38 | DNS_TTL_ANSWER | TTL of the first A record, if any |
| 39 | FTP_COMMAND_RET_CODE | FTP client command return code |
| 40 | SRC_TO_DST_IAT_MIN | Minimum inter-packet arrival time (src to dst) |
| 41 | SRC_TO_DST_IAT_MAX | Maximum inter-packet arrival time (src to dst) |
| 42 | SRC_TO_DST_IAT_AVG | Average inter-packet arrival time (src to dst) |
| 43 | SRC_TO_DST_IAT_STDDEV | Standard deviation of inter-packet arrival time (src to dst) |
| 44 | DST_TO_SRC_IAT_MIN | Minimum inter-packet arrival time (dst to src) |
| 45 | DST_TO_SRC_IAT_MAX | Maximum inter-packet arrival time (dst to src) |
| 46 | DST_TO_SRC_IAT_AVG | Average inter-packet arrival time (dst to src) |
| 47 | DST_TO_SRC_IAT_STDDEV | Standard deviation of inter-packet arrival time (dst to src) |

Non-numeric or infinite values are stored as missing and replaced by the
median of the pooled training rows before standardization.

## Labels: 22 classes

`Benign` is recognized in any letter case. Every other `Attack` value is mapped
to a shared class through `data.label_mapping` in `configs/base.yaml`; an
unmapped value stops the run with an error.

| Dataset | Raw `Attack` value(s) | Class |
|---|---|---|
| NF-UNSW-NB15-v3 | Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms | same name |
| NF-ToN-IoT-v3 | Backdoor, dos, ddos, injection, mitm, password, ransomware, scanning, xss | Backdoor, DoS, DDoS, Injection, MITM, Password, Ransomware, Scanning, XSS |
| NF-BoT-IoT-v3 | DoS, DDoS, Reconnaissance, Theft | same name |
| NF-CICIDS2018-v3 | DDOS_attack-HOIC, DDoS_attacks-LOIC-HTTP, DDOS_attack-LOIC-UDP | DDoS |
| | DoS_attacks-Hulk, -GoldenEye, -Slowloris, -SlowHTTPTest | DoS |
| | FTP-BruteForce, SSH-Bruteforce | BruteForce |
| | Brute_Force_-Web, Brute_Force_-XSS, SQL_Injection | WebAttacks |
| | Bot | Bot |
| | Infilteration (spelling as in the data) | Infiltration |

The resulting 22 classes: Benign, Analysis, Backdoor, Bot, BruteForce, DDoS,
DoS, Exploits, Fuzzers, Generic, Infiltration, Injection, MITM, Password,
Ransomware, Reconnaissance, Scanning, Shellcode, Theft, WebAttacks, Worms, XSS.
