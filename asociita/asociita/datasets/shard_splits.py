import datasets
from datasets import load_dataset
from asociita.datasets.shard_transformation import Shard_Transformation
from asociita.utils.handlers import Handler
from asociita.utils.showcase import save_random
import copy



class Shard_Splits:
    """a common class for creating various splits among the clients"""

    @staticmethod
    def random_uniform(dataset: datasets.arrow_dataset.Dataset,
                       settings: dict):
        nodes_data = []
        for shard in range(settings['shards']): # Each shard corresponds to one
            agent_data = dataset.shard(num_shards=settings['shards'], index=shard)
            
            # Shard transformation
            if shard in settings['transformations'].keys():
                if settings['save_transformations']:
                    original_imgs = copy.deepcopy(agent_data['image'])
                agent_data = Shard_Transformation.transform(agent_data, preferences=settings['transformations'][shard]) # CALL SHARD_TRANSFORMATION CLASS
                if settings['save_transformations']:
                    save_random(original_imgs, agent_data['image'], settings['transformations'][shard])

            # In-shard split between test and train data.
            agent_data = agent_data.train_test_split(test_size=settings["local_test_size"])
            nodes_data.append([agent_data['train'], agent_data['test']])
        return nodes_data
    

    @staticmethod
    def random_imbalanced(dataset: datasets.arrow_dataset.Dataset,
                          settings: dict):
        nodes_data = []
        no_agents = settings['agents']
        imbalanced_agents = settings['imbalanced_clients']
        agents = [agent for agent in range(no_agents)]
        pandas_df = dataset.to_pandas().drop('image', axis=1)
        labels = sorted(pandas_df.label.unique())

        save_distribution = False
        print_distribution = False
        
        if settings.get('save_distribution'):
            distribution_blueprint = []
            save_distribution = True
        if settings.get('print_distribution'):
            print_distribution = True
        if settings.get('custom_sample_size'):
            sample_size = settings['custom_sample_size']
        else:
            sample_size = int(len(dataset) / no_agents)
    

        # I) Sampling dominant clients
        for agent in agents:
            if agent in imbalanced_agents:
                # 1. Sampling indexes
                sampling_weights = {key: (1 - imbalanced_agents[agent][1]) / (len(labels) - 1) for key in labels}
                sampling_weights[imbalanced_agents[agent][0]] = imbalanced_agents[agent][1]
                
                # Additional step, checking the availability of every label TODO
                # required_samples = (np.array(list(sampling_weights.values())) * sample_size).astype('int')
                # counts = pandas_df['label'].value_counts()[pandas_df['label'].value_counts() > required_samples]
                
                # 2. Apllying weights
                pandas_df["weights"] = pandas_df['label'].apply(lambda x: sampling_weights[x])
                sample = pandas_df.sample(n = sample_size, weights='weights', random_state=42)
                
                counts = sample['label'].value_counts().sort_index()
                if print_distribution:
                    print(f"Distribution of client {agent} is : {counts}")
                if save_distribution:
                    ag = counts.to_dict()
                    distribution = {'agent':agent}
                    distribution.update(ag)
                    distribution_blueprint.append(distribution)
                # 3. Selecting indexes and performing test - train split.
                sampled_data = dataset.filter(lambda filter, idx: idx in list(sample.index), with_indices=True)
                agent_data = sampled_data.train_test_split(test_size=settings["local_test_size"])
                nodes_data.append([agent_data['train'], agent_data['test']])
                
                # 4. Removing samples indexes
                pandas_df.drop(sample.index, inplace=True)
            else:
                nodes_data.append([]) # Inserting placeholder to preserve in-list order.


        # II) Sampling balanced clients
        for agent in agents:
            if agent not in imbalanced_agents:
                # 1. Sampling indexes
                sample = pandas_df.sample(n = sample_size, random_state=42)
                counts = sample['label'].value_counts().sort_index()
                if print_distribution:
                    print(f"Distribution of client {agent} is : {counts}")
                if save_distribution:
                    ag = counts.to_dict()
                    distribution = {'agent':agent}
                    distribution.update(ag)
                    distribution_blueprint.append(distribution)
                # 2. Selecting indexes and performing test - train split.
                sampled_data = dataset.filter(lambda filter, idx: idx in list(sample.index), with_indices=True)
                agent_data = sampled_data.train_test_split(test_size=settings["local_test_size"])
                nodes_data[agent] = ([agent_data['train'], agent_data['test']])
                # 3. Removing samples indexes
                pandas_df.drop(sample.index, inplace=True)
        
        if save_distribution:
            Handler.save_csv_file(file = distribution_blueprint,
                                  saving_path=settings['save_path'],
                                  file_name='distribution_blueprint.csv')
        return nodes_data

    @staticmethod
    def replicate_same_dataset(dataset: datasets.arrow_dataset.Dataset,
                               settings: dict):
        nodes_data = []
        agent_data = dataset.shard(num_shards=1, index=0)
        agent_data = agent_data.train_test_split(test_size=settings["local_test_size"])
        for _ in range(settings['agents']):
            nodes_data.append([copy.deepcopy(agent_data['train']), copy.deepcopy(agent_data['test'])])
        
        return nodes_data
    

    @staticmethod
    def split_in_blocks(dataset: datasets.arrow_dataset.Dataset,
                        settings: dict):
        nodes_data = []
        for shard in range(settings['shards']):
            agent_data = dataset.shard(num_shards=settings['shards'], index=shard)
            agent_data = agent_data.train_test_split(test_size=settings["local_test_size"])
            for _ in range((int(settings['agents'] / settings['shards']))):
                nodes_data.append([copy.deepcopy(agent_data['train']), copy.deepcopy(agent_data['test'])])
        return nodes_data