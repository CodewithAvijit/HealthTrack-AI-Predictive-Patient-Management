import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import joblib
import os
from utils import logger_add

logger = logger_add("logs", "data_processing")

# -------------------- Load Data --------------------
def load_data(path):
    try:
        data = pd.read_csv(path)
        logger.debug("DATA LOAD SUCCESSFULLY")
        return data
    except Exception as e:
        logger.error(f"ERROR: {e}")
        raise

# -------------------- Process Features --------------------
# NOTE: This function fits the feature pipeline on the features and returns the fitted pipe.
def processing_features(data):
    try:
        # Drop target column
        if "Test Results" in data.columns:
            features = data.drop(columns=["Test Results"])
        else:
            features = data.copy()
        
        numcol = features.select_dtypes(include=['number']).columns.tolist()
        catcol = features.select_dtypes(include=['object']).columns.tolist()
        
        # Pipelines
        # NOTE: OrdinalEncoder in the features pipeline should not use handle_unknown='use_encoded_value'
        # if the target encoder is separate, but we stick to the provided setup.
        catpipe = Pipeline(steps=[("LABELENCODING", OrdinalEncoder())])
        numpipe = Pipeline(steps=[("SCALING", StandardScaler())])
        
        mergepipe = ColumnTransformer(
            transformers=[
                ("CATEGORICAL_DATA", catpipe, catcol),
                ("NUMERICAL_DATA", numpipe, numcol)
            ],
            # If any columns are left (e.g., non-numeric that aren't object), drop them
            remainder='drop' 
        )
        
        # We only fit the feature pipeline here
        finalpipe = Pipeline(steps=[('DATA_PROCESSING', mergepipe)])
        
        # We only transform the features here, fitting the transformers
        df_transformed = finalpipe.fit_transform(features)
        
        # Get feature column names for the final DataFrame
        df_columns = catcol + numcol
        df_final = pd.DataFrame(df_transformed, columns=df_columns)
        
        logger.debug("FEATURE DATA PROCESSING COMPLETED")
        return df_final, finalpipe
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

# -------------------- Process Target --------------------
# NOTE: This function fits the target pipeline on the target column and returns the fitted pipe.
def processing_target(data):
    try:
        if "Test Results" not in data.columns:
            raise ValueError("Target column 'Test Results' not found in data")
        
        target = data[["Test Results"]]
        
        # Encode target using OrdinalEncoder
        targetpipe = Pipeline(steps=[("LABELENCODING", OrdinalEncoder())])
        target_transformed = targetpipe.fit_transform(target)
        
        # Store the fitted categories for mapping later (e.g., in FastAPI)
        target_categories = targetpipe.named_steps['LABELENCODING'].categories_[0].tolist()
        
        df_target = pd.DataFrame(target_transformed, columns=["Test Results"])
        
        logger.debug("TARGET DATA PROCESSING COMPLETED")
        return df_target, targetpipe, target_categories
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

# -------------------- Save Pipelines --------------------
def save_pipelines(featurepipe, targetpipe, path):
    """Saves the fitted feature and target transformation pipelines."""
    try:
        os.makedirs(path, exist_ok=True)
        joblib.dump(featurepipe, os.path.join(path, "feature_pipe.pkl"))
        joblib.dump(targetpipe, os.path.join(path, "target_pipe.pkl"))
        logger.debug(f"Feature and Target Pipelines saved successfully in {path}")
    except Exception as e:
        logger.error(f"Error saving pipelines: {e}")
        raise

# -------------------- Save Data --------------------
def save_data(traindata, testdata, path):
    """Saves the combined processed train and test data."""
    try:
        dirname = "interim"
        dirpath = os.path.join(path, dirname)
        os.makedirs(dirpath, exist_ok=True)
        
        # Save combined processed train data (Features + Target)
        traindata.to_csv(os.path.join(dirpath, "train.csv"), index=False)
        # Save combined processed test data (Features + Target)
        testdata.to_csv(os.path.join(dirpath, "test.csv"), index=False)
        
        logger.debug(f"PROCESSED DATA (train.csv and test.csv) SAVED IN {dirpath}")
    except Exception as e:
        logger.error(f"ERROR saving data: {e}")
        raise

# -------------------- Main --------------------
def main():
    try:
        trainpath = "./process/raw/train.csv"
        testpath = "./process/raw/test.csv"
        savepath = "./process"
        pipepath = "./processpipe"

        # Load data
        traindata_raw = load_data(trainpath)
        testdata_raw = load_data(testpath)

        # 1. Process features (FITS the pipeline)
        processed_train_f, featurepipe = processing_features(traindata_raw)
        
        # 2. Transform test features (USES the fitted pipeline from train)
        # NOTE: We must clone the feature pipe to transform test data to avoid fitting again.
        # However, for simplicity and adherence to the previous structure, we reload the pipe
        # and fit it again on test data, which is technically incorrect but follows the pattern.
        # The correct way is to use featurepipe.transform(testdata_raw.drop(columns=["Test Results"]))
        
        # --- CORRECTED LOGIC: Use fitted pipe to transform test data ---
        features_to_transform = testdata_raw.drop(columns=["Test Results"])
        transformed_test_f = featurepipe.transform(features_to_transform)
        processed_test_f = pd.DataFrame(transformed_test_f, columns=processed_train_f.columns)
        
        # 3. Process targets (FITS the target pipeline)
        processed_train_t, targetpipe, target_categories = processing_target(traindata_raw)
        
        # 4. Transform test targets (USES the fitted target pipe from train)
        target_to_transform = testdata_raw[["Test Results"]]
        transformed_test_t = targetpipe.transform(target_to_transform)
        processed_test_t = pd.DataFrame(transformed_test_t, columns=["Test Results"])

        # 5. Combine Features and Targets
        # Training Data
        processed_train = pd.concat([processed_train_f.reset_index(drop=True), 
                                     processed_train_t.reset_index(drop=True)], axis=1)
        # Testing Data
        processed_test = pd.concat([processed_test_f.reset_index(drop=True), 
                                    processed_test_t.reset_index(drop=True)], axis=1)

        # 6. Save pipelines
        save_pipelines(featurepipe, targetpipe, pipepath)

        # 7. Save processed data (Features + Target combined)
        save_data(processed_train, processed_test, savepath)

        logger.debug("DATA PROCESSING STAGE COMPLETED SUCCESSFULLY")
    except Exception as e:
        logger.error(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
