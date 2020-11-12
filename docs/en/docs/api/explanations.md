# Explanations

**Explanations** are designed to help the end-user of a machine learning model to understand the outcome of a prediction, for example by finding out _why_  the model has given a certain result/prediction instead of another. 

Our API contains several endpoints that can be used to obtain different types of explanations for a given query, those currently implemented are reported below.

An _explanation by examples_ is a type of explanation where the user receives, given a query, the most similar instances selected from the training dataset. These examples represent the training points that the model consider most similar to the query that needs to be explained. 

This type of explanation can be used to justify the decision of a machine learning model by the fact that the model was trained on similar instances. The endpoint provides two types of examples. Pertinent examples are similar training points for which the model gives the same classification, while non pertinent examples are similar points for which the model gives a different classification. 

An _explanation by local tree_ is a type of explanation where a decision tree is used as a surrogate to reconstruct the behaviour of the model in the proximity of a given query. This kind of explanation is useful to understand how your model behaves when changing a limited number of input features, allowing, for example, for counter factual reasoning. 

Stating that the tree is constructed in the proximity of the query means that only small changes of the input vector are considered. This can lead to situations where the surrogate tree only contains one leaf: this means that no counter factual example has been found in the proximity of the query.

## GET EXPLANATION BY EXAMPLES

Return an explanation by examples for a certain prediction, given its ID. 

**URL** : `/api/explanations/byexamples/{prediction_ID}`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the ID passed as argument is valid and the prediction associated to it exists, an explanation by examples is returned.

**Code** : `200 OK`

**Content example**

```json
{
  "prediction": "prediction_id",
  "pertinent_examples": [
    {
      "age": 50,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 13,
      "native_country": "United-States"
    },
    {
      "age": 26,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 20,
      "native_country": "United-States"
    },
    {
      "age": 36,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 2407,
      "capital_loss": 0,
      "hours_per_week": 40,
      "native_country": "United-States"
    },
    {
      "age": 47,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 30,
      "native_country": "United-States"
    },
    {
      "age": 51,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 25,
      "native_country": "United-States"
    }
  ],
  "not_pertinent_examples": [
    {
      "age": 56,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 60,
      "native_country": "United-States"
    },
    {
      "age": 49,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 60,
      "native_country": "United-States"
    },
    {
      "age": 62,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 40,
      "native_country": "United-States"
    },
    {
      "age": 54,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 45,
      "native_country": "United-States"
    },
    {
      "age": 52,
      "work_class": "Self-emp-not-inc",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "White-Collar",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 50,
      "native_country": "United-States"
    }
  ],
  "timestamp": "2020-05-04T16:32:33.681930"
}

```

### Error Response

**Condition** : If provided data is invalid, e.g. the prediction ID does not exist.

**Code** : `404 NOT FOUND`

**Content example** :

```json
{
  "detail": "Prediction not found"
}
```

## GET EXPLANATION BY LOCAL TREE

Return an explanation by local tree certain prediction, given its ID. 

**URL** : `/api/explanations/bylocaltree/{prediction_ID}`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the ID passed as argument is valid and the prediction associated to it exists, an explanation by local tree is returned.

**Code** : `200 OK`

**Content example**

```json
{
  "prediction": "prediction_id",
  "tree": {
    "Feature": "Age",
    "Rule": "<= 46.551",
    "Gini": 0.5,
    "Samples": 101,
    "Split percentages": {
      "<=50k": "45.54%",
      ">50k": "54.46%"
    },
    "if True": [
      {
        "Feature": "Occupation_professional",
        "Rule": "<= 0.500",
        "Gini": 0.21,
        "Samples": 33,
        "Split percentages": {
          "<=50k": "87.88%",
          ">50k": "12.12%"
        },
        "if True": [
          {
            "Gini": 0,
            "Samples": 28,
            "Split percentages": {
              "<=50k": "100.00%",
              ">50k": "0.00%"
            }
          }
        ],
        "else if False": [
          {
            "Feature": "Capital_gain",
            "Rule": "<= 26.166",
            "Gini": 0.32,
            "Samples": 5,
            "Split percentages": {
              "<=50k": "20.00%",
              ">50k": "80.00%"
            },
            "if True": [
              {
                "Gini": 0,
                "Samples": 4,
                "Split percentages": {
                  "<=50k": "0.00%",
                  ">50k": "100.00%"
                }
              }
            ],
            "else if False": [
              {
                "Gini": 0,
                "Samples": 1,
                "Split percentages": {
                  "<=50k": "100.00%",
                  ">50k": "0.00%"
                }
              }
            ]
          }
        ]
      }
    ],
    "else if False": [
      {
        "Feature": "Occupation_blue-collar",
        "Rule": "<= 0.500",
        "Gini": 0.38,
        "Samples": 68,
        "Split percentages": {
          "<=50k": "25.00%",
          ">50k": "75.00%"
        },
        "if True": [
          {
            "Feature": "Education_high school grad",
            "Rule": "<= 0.500",
            "Gini": 0.31,
            "Samples": 63,
            "Split percentages": {
              "<=50k": "19.05%",
              ">50k": "80.95%"
            },
            "if True": [
              {
                "Gini": 0.19,
                "Samples": 55,
                "Split percentages": {
                  "<=50k": "10.91%",
                  ">50k": "89.09%"
                }
              }
            ],
            "else if False": [
              {
                "Gini": 0.38,
                "Samples": 8,
                "Split percentages": {
                  "<=50k": "75.00%",
                  ">50k": "25.00%"
                }
              }
            ]
          }
        ],
        "else if False": [
          {
            "Gini": 0,
            "Samples": 5,
            "Split percentages": {
              "<=50k": "100.00%",
              ">50k": "0.00%"
            }
          }
        ]
      }
    ]
  },
  "timestamp": "2020-05-04T16:31:27.454733"
}

```

### Error Response

**Condition** : If provided data is invalid, e.g. the prediction ID does not exist.

**Code** : `404 NOT FOUND`

**Content example** :

```json
{
  "detail": "Prediction not found"
}
```