from asociita.utils.computations import Subsets

if __name__ == "__main__":
    example = [1, 2, 3, 4]
    superset = Subsets.form_superset(example)
    print(superset)