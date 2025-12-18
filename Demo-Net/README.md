Dataset Preprocessing Overview

***Use Demo Net Folder as the environment***

This project uses three datasets for training Demo-Net and baseline GNN models:

Facebook Page–to–Page Graph

USA Airport Connectivity Graph

ENZYMES Dataset

Each dataset goes through a small preprocessing pipeline that converts raw graphs or files into PyTorch Geometric–ready tensors stored as processed.pt.
These processed files are what the training scripts load during your demo.

Facebook Dataset (facebook_paper_degX_binsY)
Folder used for training
data/facebook_paper_degX_binsY/
    processed.pt

Raw files originally used

facebook_combined.txt (or facebook348.edgelist depending on variant)

Purpose

The goal is to transform the Facebook page–to–page graph into:

Degree-based node features (degX)

Binned degree labels or classification targets (binsY)

Preprocessing Steps

Performed by:

prep_facebook_binned_degree.py

prep_facebook_degfeat_binslabels.py

prep_facebook_degree_labels.py

The scripts do the following:

Load the raw edge list and construct an undirected graph.

Compute degree for every node.

Create X (features) based on node degree or degree-derived statistics.

Create Y (labels) by binning degrees into categories.

Save everything into a PyTorch Geometric data object → processed.pt.

Training Commands
python train_nodeclf.py --model gcn --data_path data/facebook_paper_degX_binsY/processed.pt
python train_nodeclf.py --model demo_net_weight --data_path data/facebook_paper_degX_binsY/processed.pt
python train_nodeclf.py --model demo_net_hash --data_path data/facebook_paper_degX_binsY/processed.pt

🇺🇸 USA Airports Dataset (usa_airports_degX_binsY)
Folder used for training
data/usa_airports_degX_binsY/
    processed.pt

Raw files originally used

usa_airports.txt (graph structure)

Additional attributes (activity level, etc.)

Purpose

Transform the airport connectivity network into:

Degree-based node features

Activity‐bin node labels (binsY)

Preprocessing Steps

Performed by:

prep_usa_airports.py

prep_usa_airports_feats.py

prep_usa_airports_activity_labels.py

Pipeline:

Load airport graph (directed or undirected depending on preprocessing script).

Compute degree features → stored in X.

Load airport metadata (activity levels, categories).

Bin activity levels into Y labels.

Generate PyTorch Geometric Data object → processed.pt.

Training Commands
python USA_train_nodeclf.py --model gcn --data_path data/usa_airports_degX_binsY/processed.pt
python USA_train_nodeclf.py --model demo_weight --data_path data/usa_airports_degX_binsY/processed.pt
python USA_train_nodeclf.py --model demo_hash --data_path data/usa_airports_degX_binsY/processed.pt

🧬 ENZYMES Dataset (ENZYMES_raw → processed internally)
Folder used for training
data/ENZYMES_raw/
    (raw graph files for 600 enzyme graphs)

Purpose

Load the 6-class ENZYMES graph classification dataset used in GNN benchmark papers.

Preprocessing Steps

Handled inside the training script:

train_enzymes.py

Pipeline:

Load raw ENZYMES graphs using standard TUDataset/torch-geometric logic.

Apply node feature normalization when needed.

Format each graph with:

a node feature matrix

an adjacency list

the graph-level class label

Cache processed graphs for future runs.

Training Commands
python train_enzymes.py --model deepwl
python train_enzymes.py --model demo_hash
python train_enzymes.py --model demo_weight

✅ Summary Table

Dataset	Raw Folder	    Processed Folder	    Preprocessing Scripts	        Training Script
Facebook	            facebook_combined.txt	facebook_paper_degX_binsY	    prep_facebook_*	train_nodeclf.py
USA Airports	        usa_airports files	    usa_airports_degX_binsY	        prep_usa_airports_*	USA_train_nodeclf.py
ENZYMES	ENZYMES_raw	    processed by PyG	    internal to train_enzymes.py	train_enzymes.py