
import datasets

def check_intersections(train_ds, test_ds, column_name):

    train_values = set(train_ds[column_name])
    test_values = set(test_ds[column_name])

    print(column_name, "intersection", len(test_values.intersection(train_values)))

if __name__ == '__main__':

    train_ds = datasets.load_dataset('marsianin500/Speech2Latex', split='equations_train')
    test_ds = datasets.load_dataset('marsianin500/Speech2Latex', split='equations_test')

    print("Equations")
    check_intersections(train_ds, test_ds, 'sentence')
    check_intersections(train_ds, test_ds, 'sentence_normalized')

    print("\n\n")

    train_ds = datasets.load_dataset('marsianin500/Speech2Latex', split='sentences_train')
    test_ds = datasets.load_dataset('marsianin500/Speech2Latex',  split='sentences_test')

    print("Sentences")
    check_intersections(train_ds, test_ds, 'sentence')
    check_intersections(train_ds, test_ds, 'sentence_normalized')