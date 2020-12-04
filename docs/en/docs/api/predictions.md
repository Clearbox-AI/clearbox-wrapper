# Predictions

A **prediction** represents a single performed inference on a given input, using the machine learning model provided by the user. When a prediction is made using our API, it is saved in a persistent database so that it can be used later to receive explanations on its outcome and user feedback. Furthermore, for each prediction the engine assigns an _ID_, a _trust score_ and an _anomaly score_.

The _anomaly score_ is an index from 0 to 3, the higher this index the higher the probability that the input provided is distant from the training data of your dataset and therefore it could be an outlier. The _trust score_ is an index from 0 to 3, the higher is the index the higher is the prediction trustworthy, on the contrary if the index is low it means that the prediction is too uncertain to be considered reliable.

Anomaly Score | Meaning
---------|----------
 0 | The provided input is very similar to the instances on which the model was trained.  
 1 | The provided input is similar to the instances on which the model was trained. It could present some values that slightly differ from the model training dataset.
 2 | The provided input has some features that differ from the instances on which the model was trained. You should check if it contains some errors.
 3 | The provided input has a large number of features with values very different from the instances on which the model was trained. It is likelihood that it presents some anomalies or that it is an outlier.  


Trust Score | Meaning
---------|----------
 0 | The model prediction is very uncertain, similar inputs were predicted with a different label.
 1 | The model prediction is uncertain, it could have predicted a different label for the same instance as well.
 2 | The model prediction is consistent, it would be very unlikely that similar inputs could have been predicted with a different label.
 3 | The model prediction is very consistent.

## POST PREDICTION

Receive an input from the user and perform an inference on it, returning the _predicted class_, the _anomaly score_ and the _trust score_, together with an _ID_.

**URL** : `/api/predictions/`

**Method** : `POST`

**Auth required** : None

**Permissions required** : None

**Data constraints**

The input provided must follow the constraints of your dataset.

```json
{
  "age": number,
  "work_class": string,
  "education": string,
  "marital_status": string,
  "occupation": string,
  "relationship": string,
  "race": string,
  "sex": string,
  "capital_gain": number,
  "capital_loss": number,
  "hours_per_week": number,
  "native_country": string
}
```

**Data example** 

All fields must be sent.

```json
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
}
```

### Success Response

**Condition** : If everything is OK a new prediction is generated, returned to the user and stored into the database.

**Code** : `200 OK`

**Content example**

```json
{
  "input": {
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
  "output": {
    "predicted_label": "<=50K",
    "predicted_class": 0,
    "predicted_probabilities": [
      0.5781863927841187,
      0.42181360721588135
    ],
    "anomaly_score": 1,
    "trust_score": 306.4525445750764
  },
  "timestamp": "2020-04-10T08:25:21.682150",
  "_id": "5e902d7168e8ad2a707f604f"
}

```

### Error Response

**Condition** : If the data constraints are not respected.

**Code** : `422 UNPROCESSABLE ENTITY`

**Content example** :

```json
{
  "detail": [
    {
      "loc": [
        "body",
        "data",
        "work_class"
      ],
      "msg": "Column 'work_class': 'Student' is not a valid value. Allowed values: {'Local-gov', 'Self-emp-inc', 'Without-pay', 'Self-emp-not-inc', 'Federal-gov', 'Private', 'State-gov'}.",
      "type": "value_error"
    },
    ...
  ]
}
```

## GET PREDICTIONS

Return all the inferred predictions in chronological order starting from the last one performed. 

**URL** : `/api/predictions/`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Parameters

**Last** : If specified, returns only the last _n_ predictions made.

**Since** : If specified (following the format _%m/%d/%Y_), returns only the predictions made since that date.

### Success Response

**Condition** : If everything is OK a the requested predictions are returned to the user.

**Code** : `200 OK`

**Content example**

```json
[
  {
    "input": {
      "age": 0,
      "work_class": "string",
      "education": "string",
      "marital_status": "string",
      "occupation": "string",
      "relationship": "string",
      "race": "string",
      "sex": "string",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 0,
      "native_country": "string"
    },
    "output": {
      "predicted_label": "string",
      "anomaly_score": 0,
      "trust_score": 0
    },
    "timestamp": "2020-04-10T09:00:33.568Z",
    "_id": "string"
  }
]

```

### Error Response

**Condition** : If no prediction has been found in the database.

**Code** : `404 NOT FOUND`

**Content example** :

```json
{
  "detail": "No predictions found"
}
```

## GET SINGLE PREDICTION

Return a single performed prediction, given its ID. 

**URL** : `/api/predictions/{prediction_ID}`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the ID passed as argument is valid and the prediction associated to it exists, the prediction will be returned to the user.

**Code** : `200 OK`

**Content example**

```json
{
  "input": {
    "age": 0,
    "work_class": "string",
    "education": "string",
    "marital_status": "string",
    "occupation": "string",
    "relationship": "string",
    "race": "string",
    "sex": "string",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 0,
    "native_country": "string"
  },
  "output": {
    "predicted_label": "string",
    "anomaly_score": 0,
    "trust_score": 0
  },
  "timestamp": "2020-04-10T09:00:33.568Z",
  "_id": "string"
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

## DELETE SINGLE PREDICTION

Delete a single performed prediction, given its ID. 

**URL** : `/api/predictions/{prediction_ID}`

**Method** : `DELETE`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the ID passed as argument is valid and the prediction associated to it exists, the prediction will be deleted from the database and returned to the user.

**Code** : `200 OK`

**Content example**

```json
{
  "input": {
    "age": 0,
    "work_class": "string",
    "education": "string",
    "marital_status": "string",
    "occupation": "string",
    "relationship": "string",
    "race": "string",
    "sex": "string",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 0,
    "native_country": "string"
  },
  "output": {
    "predicted_label": "string",
    "anomaly_score": 0,
    "trust_score": 0
  },
  "timestamp": "2020-04-10T09:00:33.568Z",
  "_id": "string"
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