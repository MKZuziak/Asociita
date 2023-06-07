import datasets

def save_blueprint(dataset: list[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset,], path:str):
    with open(path, 'a+', ) as csv_file:
        orchestrator_data = dataset[0]
        nodes_data = dataset[1]
        
        # WRITE HEADERS
        header = ["client_id", "partition", "total_samples"]
        labels = nodes_data[0][0].features['label'].names
        header.extend(labels)
        csv_file.write(",".join(header) + '\n')

        # WRITE ORCHESTRATOR
        row = ['orchestrator', 'central_test_set', str(len(orchestrator_data))]
        for label in labels:
            row.append(str(len(orchestrator_data.filter(lambda inst: inst['label'] == int(label)))))
        csv_file.write(",".join(row) + '\n')

        # WRITE CLIENTS
        for client, data in enumerate(nodes_data):
            row = [str(client), 'train_set', str(len(data[0]))]
            for label in labels:
                row.append(str(len(data[0].filter(lambda inst: inst['label'] == int(label)))))
            csv_file.write(",".join(row) + '\n')

            row = [str(client), 'test_set', str(len(data[1]))]
            for label in labels:
                row.append(str(len(data[1].filter(lambda inst: inst['label'] == int(label)))))
            csv_file.write(",".join(row) + '\n')


