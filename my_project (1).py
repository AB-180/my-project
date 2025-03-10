{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "444195d5-ce88-4986-9f26-c03a9fde54da",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv('music_sentiment_dataset.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6034ad34-b3cc-473e-8d96-094484147609",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  User_ID                                          User_Text Sentiment_Label  \\\n",
      "0      U1  Way ball purpose public experience recently re...             Sad   \n",
      "1      U2                         Save officer two myself a.           Happy   \n",
      "2      U3  Decade ahead everyone environment themselves a...         Relaxed   \n",
      "3      U4  Best change letter citizen try ask quality pro...           Happy   \n",
      "4      U5                Worker player chance kind actually.           Happy   \n",
      "\n",
      "  Recommended_Song_ID         Song_Name             Artist      Genre  \\\n",
      "0                  S1  Someone Like You              Adele        Pop   \n",
      "1                  S2             Happy  Pharrell Williams        Pop   \n",
      "2                  S3     Clair de Lune            Debussy  Classical   \n",
      "3                  S4             Happy  Pharrell Williams        Pop   \n",
      "4                  S5             Happy  Pharrell Williams        Pop   \n",
      "\n",
      "   Tempo (BPM)         Mood Energy Danceability  \n",
      "0           67  Melancholic    Low          Low  \n",
      "1          160       Joyful   High         High  \n",
      "2           60     Soothing    Low          Low  \n",
      "3          160       Joyful   High         High  \n",
      "4          160       Joyful   High         High  \n"
     ]
    }
   ],
   "source": [
    "# Inspect the data\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "40044ed4-433d-4570-ab5e-db5531a62434",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Encoded Energy: [1 0]\n",
      "Encoded Danceability: [1 0 2]\n",
      "  User_ID                                          User_Text  \\\n",
      "0      U1  Way ball purpose public experience recently re...   \n",
      "1      U2                         Save officer two myself a.   \n",
      "2      U3  Decade ahead everyone environment themselves a...   \n",
      "3      U4  Best change letter citizen try ask quality pro...   \n",
      "4      U5                Worker player chance kind actually.   \n",
      "\n",
      "  Recommended_Song_ID         Song_Name             Artist  Tempo (BPM)  \\\n",
      "0                  S1  Someone Like You              Adele           67   \n",
      "1                  S2             Happy  Pharrell Williams          160   \n",
      "2                  S3     Clair de Lune            Debussy           60   \n",
      "3                  S4             Happy  Pharrell Williams          160   \n",
      "4                  S5             Happy  Pharrell Williams          160   \n",
      "\n",
      "   Energy  Danceability  Genre_Ambient  Genre_Classical  ...  Mood_Emotional  \\\n",
      "0       1             1          False            False  ...           False   \n",
      "1       0             0          False            False  ...           False   \n",
      "2       1             1          False             True  ...           False   \n",
      "3       0             0          False            False  ...           False   \n",
      "4       0             0          False            False  ...           False   \n",
      "\n",
      "   Mood_Energetic  Mood_Joyful  Mood_Melancholic  Mood_Powerful  \\\n",
      "0           False        False              True          False   \n",
      "1           False         True             False          False   \n",
      "2           False        False             False          False   \n",
      "3           False         True             False          False   \n",
      "4           False         True             False          False   \n",
      "\n",
      "   Mood_Soothing  Sentiment_Label_Happy  Sentiment_Label_Motivated  \\\n",
      "0          False                  False                      False   \n",
      "1          False                   True                      False   \n",
      "2           True                  False                      False   \n",
      "3          False                   True                      False   \n",
      "4          False                   True                      False   \n",
      "\n",
      "   Sentiment_Label_Relaxed  Sentiment_Label_Sad  \n",
      "0                    False                 True  \n",
      "1                    False                False  \n",
      "2                     True                False  \n",
      "3                    False                False  \n",
      "4                    False                False  \n",
      "\n",
      "[5 rows x 25 columns]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Initialize label encoders\n",
    "energy_encoder = LabelEncoder()\n",
    "danceability_encoder = LabelEncoder()\n",
    "\n",
    "# Fit label encoders on all possible values\n",
    "energy_encoder.fit(df['Energy'])\n",
    "danceability_encoder.fit(df['Danceability'])\n",
    "\n",
    "# Encode 'Energy' and 'Danceability' columns\n",
    "df['Energy'] = energy_encoder.transform(df['Energy'])\n",
    "df['Danceability'] = danceability_encoder.transform(df['Danceability'])\n",
    "\n",
    "# Inspect the encoded columns\n",
    "print(\"Encoded Energy:\", df['Energy'].unique())\n",
    "print(\"Encoded Danceability:\", df['Danceability'].unique())\n",
    "\n",
    "# Perform one-hot encoding on other categorical columns\n",
    "df_encoded = pd.get_dummies(df, columns=['Genre', 'Mood', 'Sentiment_Label'])\n",
    "\n",
    "# Inspect the encoded DataFrame\n",
    "print(df_encoded.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2bb2d662-9bf6-447b-b727-50f3ceccc3d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Features (X):\n",
      "   Tempo (BPM)  Energy  Danceability  Genre_Ambient  Genre_Classical  \\\n",
      "0           67       1             1          False            False   \n",
      "1          160       0             0          False            False   \n",
      "2           60       1             1          False             True   \n",
      "3          160       0             0          False            False   \n",
      "4          160       0             0          False            False   \n",
      "\n",
      "   Genre_Funk  Genre_Hip-Hop  Genre_Pop  Genre_Rock  Mood_Calm  \\\n",
      "0       False          False       True       False      False   \n",
      "1       False          False       True       False      False   \n",
      "2       False          False      False       False      False   \n",
      "3       False          False       True       False      False   \n",
      "4       False          False       True       False      False   \n",
      "\n",
      "   Mood_Emotional  Mood_Energetic  Mood_Joyful  Mood_Melancholic  \\\n",
      "0           False           False        False              True   \n",
      "1           False           False         True             False   \n",
      "2           False           False        False             False   \n",
      "3           False           False         True             False   \n",
      "4           False           False         True             False   \n",
      "\n",
      "   Mood_Powerful  Mood_Soothing  Sentiment_Label_Happy  \\\n",
      "0          False          False                  False   \n",
      "1          False          False                   True   \n",
      "2          False           True                  False   \n",
      "3          False          False                   True   \n",
      "4          False          False                   True   \n",
      "\n",
      "   Sentiment_Label_Motivated  Sentiment_Label_Relaxed  Sentiment_Label_Sad  \n",
      "0                      False                    False                 True  \n",
      "1                      False                    False                False  \n",
      "2                      False                     True                False  \n",
      "3                      False                    False                False  \n",
      "4                      False                    False                False  \n",
      "\n",
      "Target (y):\n",
      "0    S1\n",
      "1    S2\n",
      "2    S3\n",
      "3    S4\n",
      "4    S5\n",
      "Name: Recommended_Song_ID, dtype: object\n"
     ]
    }
   ],
   "source": [
    "# Define features (X) - all columns except non-feature columns\n",
    "X = df_encoded.drop(['User_ID', 'User_Text', 'Recommended_Song_ID', 'Song_Name', 'Artist'], axis=1)\n",
    "\n",
    "# Define target (y) - the column we want to predict\n",
    "y = df_encoded['Recommended_Song_ID']\n",
    "\n",
    "# Inspect the features and target\n",
    "print(\"Features (X):\")\n",
    "print(X.head())\n",
    "print(\"\\nTarget (y):\")\n",
    "print(y.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "48642c0d-6e3a-42d8-8b13-5fcc2d0742ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set shape: (800, 20) (800,)\n",
      "Testing set shape: (200, 20) (200,)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Split the data into training (80%) and testing (20%) sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Inspect the shapes of the splits\n",
    "print(\"Training set shape:\", X_train.shape, y_train.shape)\n",
    "print(\"Testing set shape:\", X_test.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "bdcbc362-1b17-4a71-8596-59b597c5a1d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;RandomForestClassifier<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html\">?<span>Documentation for RandomForestClassifier</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>RandomForestClassifier(random_state=42)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(random_state=42)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Initialize the model\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "\n",
    "# Train the model on the training data\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f3050c85-c139-4aeb-91ff-0530d4687a1a",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'recommended_song' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[11], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# Decode Energy and Danceability\u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m recommended_song[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mEnergy\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m=\u001b[39m energy_encoder\u001b[38;5;241m.\u001b[39minverse_transform([recommended_song[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mEnergy\u001b[39m\u001b[38;5;124m'\u001b[39m]])[\u001b[38;5;241m0\u001b[39m]\n\u001b[1;32m      3\u001b[0m recommended_song[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mDanceability\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m=\u001b[39m danceability_encoder\u001b[38;5;241m.\u001b[39minverse_transform([recommended_song[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mDanceability\u001b[39m\u001b[38;5;124m'\u001b[39m]])[\u001b[38;5;241m0\u001b[39m]\n",
      "\u001b[0;31mNameError\u001b[0m: name 'recommended_song' is not defined"
     ]
    }
   ],
   "source": [
    "# Decode Energy and Danceability\n",
    "recommended_song['Energy'] = energy_encoder.inverse_transform([recommended_song['Energy']])[0]\n",
    "recommended_song['Danceability'] = danceability_encoder.inverse_transform([recommended_song['Danceability']])[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d91e86e5-b213-439d-9abc-ab96a4a1876c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_recommendation(user_input):\n",
    "    # Convert user input into a DataFrame\n",
    "    user_df = pd.DataFrame([user_input])\n",
    "\n",
    "    # Check if user input values are valid\n",
    "    valid_energy_values = energy_encoder.classes_\n",
    "    valid_danceability_values = danceability_encoder.classes_\n",
    "\n",
    "    if user_df['Energy'].iloc[0] not in valid_energy_values:\n",
    "        raise ValueError(f\"Invalid Energy value. Valid values are: {valid_energy_values}\")\n",
    "    if user_df['Danceability'].iloc[0] not in valid_danceability_values:\n",
    "        raise ValueError(f\"Invalid Danceability value. Valid values are: {valid_danceability_values}\")\n",
    "\n",
    "    # Encode 'Energy' and 'Danceability' columns\n",
    "    user_df['Energy'] = energy_encoder.transform([user_df['Energy']])[0]\n",
    "    user_df['Danceability'] = danceability_encoder.transform([user_df['Danceability']])[0]\n",
    "\n",
    "    # Perform one-hot encoding on user input\n",
    "    user_encoded = pd.get_dummies(user_df, columns=['Genre', 'Mood', 'Sentiment_Label'])\n",
    "\n",
    "    # Ensure all columns are present\n",
    "    for col in X.columns:\n",
    "        if col not in user_encoded.columns:\n",
    "            user_encoded[col] = 0\n",
    "\n",
    "    # Reorder columns to match training data\n",
    "    user_encoded = user_encoded[X.columns]\n",
    "\n",
    "    # Predict the recommended song ID\n",
    "    recommended_song_id = model.predict(user_encoded)[0]\n",
    "\n",
    "    # Get the song details\n",
    "    recommended_song = df[df['Recommended_Song_ID'] == recommended_song_id].iloc[0]\n",
    "\n",
    "    # Decode Energy and Danceability\n",
    "    recommended_song['Energy'] = energy_encoder.inverse_transform([recommended_song['Energy']])[0]\n",
    "    recommended_song['Danceability'] = danceability_encoder.inverse_transform([recommended_song['Danceability']])[0]\n",
    "\n",
    "    return recommended_song"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79d7d4e4-4e25-4b13-95be-287cdc39c919",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example user input\n",
    "user_input = {\n",
    "    'Tempo (BPM)': 120,\n",
    "    'Energy': 'High',\n",
    "    'Danceability': 'High',\n",
    "    'Genre': 'Pop',\n",
    "    'Mood': 'Joyful',\n",
    "    'Sentiment_Label': 'Happy'\n",
    "}\n",
    "\n",
    "# Get the recommended song\n",
    "try:\n",
    "    recommended_song = predict_recommendation(user_input)\n",
    "    # Display the recommended song\n",
    "    print(\"\\nRecommended Song:\")\n",
    "    print(f\"Song Name: {recommended_song['Song_Name']}\")\n",
    "    print(f\"Artist: {recommended_song['Artist']}\")\n",
    "    print(f\"Genre: {recommended_song['Genre']}\")\n",
    "    print(f\"Mood: {recommended_song['Mood']}\")\n",
    "    print(f\"Tempo (BPM): {recommended_song['Tempo (BPM)']}\")\n",
    "    print(f\"Energy: {recommended_song['Energy']}\")\n",
    "    print(f\"Danceability: {recommended_song['Danceability']}\")\n",
    "except ValueError as e:\n",
    "    print(e)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0c2ea7a-3bb4-4ff2-884d-a47dde2ebd14",
   "metadata": {},
   "outputs": [],
   "source": [
    "user_input = {\n",
    "    'Tempo (BPM)': 60,\n",
    "    'Energy': 'Low',\n",
    "    'Danceability': 'Low',\n",
    "    'Genre': 'Classical',\n",
    "    'Mood': 'Soothing',\n",
    "    'Sentiment_Label': 'Relaxed'\n",
    "}\n",
    "try:\n",
    "    recommended_song = predict_recommendation(user_input)\n",
    "    # Display the recommended song\n",
    "    print(\"\\nRecommended Song:\")\n",
    "    print(f\"Song Name: {recommended_song['Song_Name']}\")\n",
    "    print(f\"Artist: {recommended_song['Artist']}\")\n",
    "    print(f\"Genre: {recommended_song['Genre']}\")\n",
    "    print(f\"Mood: {recommended_song['Mood']}\")\n",
    "    print(f\"Tempo (BPM): {recommended_song['Tempo (BPM)']}\")\n",
    "    print(f\"Energy: {recommended_song['Energy']}\")\n",
    "    print(f\"Danceability: {recommended_song['Danceability']}\")\n",
    "except ValueError as e:\n",
    "    print(e)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8cccd3a4-92ce-42f6-8bb9-2c1084fe745b",
   "metadata": {},
   "outputs": [],
   "source": [
    "user_input = {\n",
    "    'Tempo (BPM)': 120,\n",
    "    'Energy': 'High',\n",
    "    'Danceability': 'High',\n",
    "    'Genre': 'Rock',\n",
    "    'Mood': 'Energetic',\n",
    "    'Sentiment_Label': 'Happy'\n",
    "}\n",
    "try:\n",
    "    recommended_song = predict_recommendation(user_input)\n",
    "    # Display the recommended song\n",
    "    print(\"\\nRecommended Song:\")\n",
    "    print(f\"Song Name: {recommended_song['Song_Name']}\")\n",
    "    print(f\"Artist: {recommended_song['Artist']}\")\n",
    "    print(f\"Genre: {recommended_song['Genre']}\")\n",
    "    print(f\"Mood: {recommended_song['Mood']}\")\n",
    "    print(f\"Tempo (BPM): {recommended_song['Tempo (BPM)']}\")\n",
    "    print(f\"Energy: {recommended_song['Energy']}\")\n",
    "    print(f\"Danceability: {recommended_song['Danceability']}\")\n",
    "except ValueError as e:\n",
    "    print(e)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4d0edc8-eaf3-469d-af5a-605c21c49f84",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('music_sentiment_dataset.csv')
    return df

# Preprocess the data
@st.cache_data
def preprocess_data(df):
    # Initialize label encoders
    energy_encoder = LabelEncoder()
    danceability_encoder = LabelEncoder()

    # Fit label encoders on all possible values
    energy_encoder.fit(df['Energy'])
    danceability_encoder.fit(df['Danceability'])

    # Encode 'Energy' and 'Danceability' columns
    df['Energy'] = energy_encoder.transform(df['Energy'])
    df['Danceability'] = danceability_encoder.transform(df['Danceability'])

    # Perform one-hot encoding on other categorical columns
    df_encoded = pd.get_dummies(df, columns=['Genre', 'Mood', 'Sentiment_Label'])

    # Define features (X) and target (y)
    X = df_encoded.drop(['User_ID', 'User_Text', 'Recommended_Song_ID', 'Song_Name', 'Artist'], axis=1)
    y = df_encoded['Recommended_Song_ID']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, energy_encoder, danceability_encoder, df

# Streamlit app
def main():
    st.title("Music Recommendation System")

    # Load data and model
    df = load_data()
    model, energy_encoder, danceability_encoder, _ = preprocess_data(df)

    # User input
    st.sidebar.header("User Input")
    tempo = st.sidebar.slider("Tempo (BPM)", 60, 200, 120)
    energy = st.sidebar.selectbox("Energy", ["Low", "High"])
    danceability = st.sidebar.selectbox("Danceability", ["Low", "High"])
    genre = st.sidebar.selectbox("Genre", df['Genre'].unique())
    mood = st.sidebar.selectbox("Mood", df['Mood'].unique())
    sentiment = st.sidebar.selectbox("Sentiment", df['Sentiment_Label'].unique())

    # Predict function
    def predict_recommendation(user_input):
        # Convert user input into a DataFrame
        user_df = pd.DataFrame([user_input])

        # Check if user input values are valid
        valid_energy_values = energy_encoder.classes_
        valid_danceability_values = danceability_encoder.classes_

        if user_df['Energy'].iloc[0] not in valid_energy_values:
            raise ValueError(f"Invalid Energy value. Valid values are: {valid_energy_values}")
        if user_df['Danceability'].iloc[0] not in valid_danceability_values:
            raise ValueError(f"Invalid Danceability value. Valid values are: {valid_danceability_values}")

        # Encode 'Energy' and 'Danceability' columns
        user_df['Energy'] = energy_encoder.transform([user_df['Energy']])[0]
        user_df['Danceability'] = danceability_encoder.transform([user_df['Danceability']])[0]

        # Perform one-hot encoding on user input
        user_encoded = pd.get_dummies(user_df, columns=['Genre', 'Mood', 'Sentiment_Label'])

        # Ensure all columns are present
        for col in X.columns:
            if col not in user_encoded.columns:
                user_encoded[col] = 0

        # Reorder columns to match training data
        user_encoded = user_encoded[X.columns]

        # Predict the recommended song ID
        recommended_song_id = model.predict(user_encoded)[0]

        # Get the song details
        recommended_song = df[df['Recommended_Song_ID'] == recommended_song_id].iloc[0]

        # Decode Energy and Danceability
        recommended_song['Energy'] = energy_encoder.inverse_transform([recommended_song['Energy']])[0]
        recommended_song['Danceability'] = danceability_encoder.inverse_transform([recommended_song['Danceability']])[0]

        return recommended_song

    # Button to trigger prediction
    if st.sidebar.button("Recommend Song"):
        user_input = {
            'Tempo (BPM)': tempo,
            'Energy': energy,
            'Danceability': danceability,
            'Genre': genre,
            'Mood': mood,
            'Sentiment_Label': sentiment
        }
        try:
            recommended_song = predict_recommendation(user_input)
            st.success("Recommended Song:")
            st.write(f"Song Name: {recommended_song['Song_Name']}")
            st.write(f"Artist: {recommended_song['Artist']}")
            st.write(f"Genre: {recommended_song['Genre']}")
            st.write(f"Mood: {recommended_song['Mood']}")
            st.write(f"Tempo (BPM): {recommended_song['Tempo (BPM)']}")
            st.write(f"Energy: {recommended_song['Energy']}")
            st.write(f"Danceability: {recommended_song['Danceability']}")
        except ValueError as e:
            st.error(e)

# Run the app
if __name__ == "__main__":
    main()
    
