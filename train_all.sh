python train.py --dataset titanic --preprocessed data/titanic_cleaned.csv --target Survived --alg logistic --task classification
python train.py --dataset titanic --preprocessed data/titanic_cleaned.csv --target Survived --alg decision_tree --task classification
python train.py --dataset titanic --preprocessed data/titanic_cleaned.csv --target Survived --alg rf --task classification

python train.py --dataset zoo --preprocessed data/zoo_data-classification.csv --target animal_name --alg logistic --task classification
python train.py --dataset zoo --preprocessed data/zoo_data-classification.csv --target animal_name --alg decision_tree --task classification
python train.py --dataset zoo --preprocessed data/zoo_data-classification.csv --target animal_name --alg rf --task classification

python train.py --dataset salary_data --preprocessed data/Salary_Data.csv --target Salary --alg linear --task regression
python train.py --dataset salary_data --preprocessed data/Salary_Data.csv --target Salary --alg ridge --task regression
python train.py --dataset salary_data --preprocessed data/Salary_Data.csv --target Salary --alg lasso --task regression
python train.py --dataset salary_data --preprocessed data/Salary_Data.csv --target Salary --alg rf --task regression

python train.py --dataset insurance --preprocessed data/insurance_cleaned.csv --target expenses --alg linear --task regression
python train.py --dataset insurance --preprocessed data/insurance_cleaned.csv --target expenses --alg ridge --task regression
python train.py --dataset insurance --preprocessed data/insurance_cleaned.csv --target expenses --alg lasso --task regression
python train.py --dataset insurance --preprocessed data/insurance_cleaned.csv --target expenses --alg rf --task regression
