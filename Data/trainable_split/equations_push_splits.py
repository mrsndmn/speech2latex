
from datasets import load_dataset, Dataset, DatasetDict


if __name__ == "__main__":
    dataset = load_dataset('marsianin500/Speech2Latex', num_proc=16)

    dev_new_equations = Dataset.load_from_disk('equations_dev_new')
    test_new_equations = Dataset.load_from_disk('equations_test_new')

    assert len(dataset['equations']) == len(dev_new_equations) + len(test_new_equations)

    all_equations_test = set(test_new_equations['sentence'])

    for dev_equation in dev_new_equations['sentence']:
        if dev_equation in all_equations_test:
            raise ValueError(f'dev_equation {dev_equation} is in test_new_equations')

    dataset['equations_train'] = dev_new_equations
    dataset['equations_test'] = test_new_equations

    dataset.push_to_hub('marsianin500/Speech2Latex', num_proc=16)