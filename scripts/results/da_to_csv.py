import json
from pathlib import Path
import csv

models_dir = Path("../DA_logs")
for metric in ["AUC", "PR"]:
    for endpoint in models_dir.iterdir():
        if endpoint.is_dir():
            with open(models_dir / f'{endpoint.name}_{metric}_results.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                for model in endpoint.iterdir():
                    if model.name == "LSTM":
                        continue
                    for target in model.iterdir():
                        for target_size in target.iterdir():
                            with open(target_size / 'averaged_source_metrics.json', 'r') as f:
                                results = json.load(f)
                                writer.writerow([model.name, target.name, target_size.name] + [source_metrics[metric] for source_name, source_metrics in results.items()])
