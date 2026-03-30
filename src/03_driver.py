from preprocessing import preprocess_zone_data
from training_adapters import train_adapters
from training_global import train_global
from evaluation import evaluate


# -------------------------
# TRAIN
# -------------------------
train_zone_data, train_labels, zones = preprocess_zone_data(
    "../inputs/combined",
    seq_len=10
)

adapters = train_adapters(train_zone_data, train_labels, zones)
global_model = train_global(train_zone_data, train_labels, zones, adapters)


# -------------------------
# TEST
# -------------------------
test_zone_data, test_labels, _ = preprocess_zone_data(
    "../inputs/test_data",
    seq_len=10
)

evaluate(test_zone_data, test_labels, zones)