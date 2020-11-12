# Dataset

A **dataset** contains all the original and unprocessed data on which the machine learning model was fitted. 

It is a tabular dataset suitable for supervised learning and classification tasks, so there are several X columns
and a single y target column. Both ordinal and categorical columns are allowed.
## INFO

Return name, timestamp and shape of the loaded dataset.

**URL** : `/api/dataset/info`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the loaded dataset is valid, the dataset info are returned.

**Code** : `200 OK`

**Content example**

```json
{
  "name": "Adult Training Dataset",
  "timestamp": "2020-04-03T17:07:14.488753",
  "rows": 30162,
  "columns": 13
}
```

### Error Response

**Condition** : If the loaded dataset is not valid.

**Code** : `422 UNPROCESSABLE ENTITY`

**Content example** :

```json
{
  "detail": "Unprocessable entity."
}
```

## COLUMNS

Return the column names of the loaded dataset.

**URL** : `/api/dataset/columns`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the loaded dataset is valid, the dataset column names are returned.

**Code** : `200 OK`

**Content example**

```json
{
  "columns": [
    "age",
    "work_class",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
  ]
}
```

### Error Response

**Condition** : If the loaded dataset is not valid.

**Code** : `422 UNPROCESSABLE ENTITY`

**Content example** :

```json
{
  "detail": "Unprocessable entity."
}
```

## BOUNDS

Return the bounds for each column of the loaded dataset.

For a numeric column the bounds are returned as a dictionary `{'max': max_value, 'min': min_value}` and 
for a categorical column the bounds are returned as dictionary of allowed values `{allowed_value+}`.

**URL** : `/api/dataset/bounds`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the loaded dataset is valid, the dataset bounds are returned.

**Code** : `200 OK`

**Content example**

```json
{
  "age": {
    "min": 17,
    "max": 90
  },
  "capital_gain": {
    "min": 0,
    "max": 99999
  },
  "capital_loss": {
    "min": 0,
    "max": 4356
  },
  "hours_per_week": {
    "min": 1,
    "max": 99
  },
  "work_class": [
    "Self-emp-inc",
    "Self-emp-not-inc",
    "Local-gov",
    "Federal-gov",
    "Without-pay",
    "Private",
    "State-gov"
  ],
  "education": [
    "Prof-School",
    "Dropout",
    "Bachelors",
    "Associates",
    "Doctorate",
    "Masters",
    "High School grad"
  ],
  "marital_status": [
    "Separated",
    "Never-Married",
    "Married",
    "Widowed"
  ],
  "occupation": [
    "Other",
    "Service",
    "Sales",
    "White-Collar",
    "Military",
    "Blue-Collar",
    "Admin",
    "Professional"
  ],
  "relationship": [
    "Wife",
    "Own-child",
    "Not-in-family",
    "Unmarried",
    "Other-relative",
    "Husband"
  ],
  "race": [
    "Other",
    "Black",
    "Amer-Indian-Eskimo",
    "White",
    "Asian-Pac-Islander"
  ],
  "sex": [
    "Female",
    "Male"
  ],
  "native_country": [
    "Other",
    "British-Commonwealth",
    "United-States",
    "SE-Asia",
    "South-America",
    "China",
    "Yugoslavia",
    "Latin-America",
    "Euro_1",
    "Euro_2"
  ],
  "income": [
    ">50K",
    "<=50K"
  ]
}
```

### Error Response

**Condition** : If the loaded dataset is not valid.

**Code** : `422 UNPROCESSABLE ENTITY`

**Content example** :

```json
{
  "detail": "Unprocessable entity."
}
```

## HEAD

Return the first 5 records of the loaded dataset as a list of dictionaries.

**URL** : `/api/dataset/head`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the loaded dataset is valid, the first records are returned.

**Code** : `200 OK`

**Content example**

```json
{
  "samples": [
    {
      "age": 39,
      "work_class": "State-gov",
      "education": "Bachelors",
      "marital_status": "Never-Married",
      "occupation": "Admin",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "capital_gain": 2174,
      "capital_loss": 0,
      "hours_per_week": 40,
      "native_country": "United-States",
      "income": "<=50K"
    },
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
      "native_country": "United-States",
      "income": "<=50K"
    },
    {
      "age": 38,
      "work_class": "Private",
      "education": "High School grad",
      "marital_status": "Separated",
      "occupation": "Blue-Collar",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 40,
      "native_country": "United-States",
      "income": "<=50K"
    },
    {
      "age": 53,
      "work_class": "Private",
      "education": "Dropout",
      "marital_status": "Married",
      "occupation": "Blue-Collar",
      "relationship": "Husband",
      "race": "Black",
      "sex": "Male",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 40,
      "native_country": "United-States",
      "income": "<=50K"
    },
    {
      "age": 28,
      "work_class": "Private",
      "education": "Bachelors",
      "marital_status": "Married",
      "occupation": "Professional",
      "relationship": "Wife",
      "race": "Black",
      "sex": "Female",
      "capital_gain": 0,
      "capital_loss": 0,
      "hours_per_week": 40,
      "native_country": "Other",
      "income": "<=50K"
    }
  ]
}
```

### Error Response

**Condition** : If the loaded dataset is not valid.

**Code** : `422 UNPROCESSABLE ENTITY`

**Content example** :

```json
{
  "detail": "Unprocessable entity."
}
```

## STATS

Return stats and values distribution of a column of the loaded dataset.

**URL** : `/api/dataset/stats`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Parameters

**column_name**, required : a valid column name of the loaded dataset.

### Success Response

**Condition** : If the column name is a valid column of the loaded dataset, the column 
stats and values distribution are returned.

**Code** : `200 OK`

**Content example**

```json
{
  "column_stats": {
    "count": 30162,
    "unique": 8,
    "top": "Blue-Collar",
    "freq": 9907
  },
  "values_dist": {
    "Blue-Collar": {
      "count": 9907,
      "freq": 32.85
    },
    "Professional": {
      "count": 4038,
      "freq": 13.39
    },
    "White-Collar": {
      "count": 3992,
      "freq": 13.24
    },
    "Admin": {
      "count": 3721,
      "freq": 12.34
    },
    "Sales": {
      "count": 3584,
      "freq": 11.88
    },
    "Service": {
      "count": 3355,
      "freq": 11.12
    },
    "Other": {
      "count": 1556,
      "freq": 5.16
    },
    "Military": {
      "count": 9,
      "freq": 0.03
    }
  }
}
```

### Error Response

**Condition** : If the column_name passed as a parameter is not a column of the loaded dataset.

**Code** : `422 UNPROCESSABLE ENTITY`

**Content example** :

```json
{
  "detail": "'column_name' is not a column of the loaded dataset."
}
```

## VARIANCE

Return the unbiased variance for each column of the loaded dataset, normalized by N by default.

**URL** : `/api/dataset/variance`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the loaded dataset is valid, the variances of the columns are returned.

**Code** : `200 OK`

**Content example**

```json
{
  "capital_gain": 54852149.7839819,
  "capital_loss": 163451.75306970818,
  "age": 172.5136990398044,
  "hours_per_week": 143.51526382775054,
  "occupation": 6.109364981338287,
  "relationship": 2.564199117485281,
  "education": 2.2797619596448677,
  "native_country": 1.8581749483994452,
  "work_class": 0.9099418540712512,
  "marital_status": 0.7073272104657116,
  "race": 0.6967167253288419,
  "sex": 0.21913490856898,
  "income": 0.18696008158506988
}
```

### Error Response

**Condition** : If the loaded dataset is not valid.

**Code** : `422 UNPROCESSABLE ENTITY`

**Content example** :

```json
{
  "detail": "Unprocessable entity."
}
```