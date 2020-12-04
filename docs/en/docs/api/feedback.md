# Feedback

The **Feedback** endpoint can be used by the user to specify whether a certain prediction was correct or not. In case of wrong prediction the user can indicate what the correct label should have been. 

Feedback are stored in a persistent database. They are used by our application to provide monitoring metrics, to calculate real time uncertainties and to perform active learning.

**DISCLAIMER:** This is an experimental feature at the moment. The collected feedback will be used in future releases that will include monitoring and uncertainty analysis.

## POST FEEDBACK

Post a feedback provided by the user, which consists of the ID of the prediction referred to, a field to specify whether the latter is correct or not and in case the label that should have been predicted.

**URL** : `/api/feedback/`

**Method** : `POST`

**Auth required** : None

**Permissions required** : None

**Data constraints**

The input provided must follow the constraints specified above.

```json
{
  "prediction": "string",
  "correctness": true,
  "new_label": "optional string"
}
```

**Data example** 

All fields must be sent.

```json
{
  "prediction": "5e902d7168e8ad2a707f604f",
  "correctness": false,
  "new_label": ">50K"
}
```

### Success Response

**Condition** : If everything is OK the new feedback is stored into the database.

**Code** : `200 OK`

**Content example**

```json
{
  "prediction": "5e902d7168e8ad2a707f604f",
  "correctness": false,
  "new_label": ">50K"
  "timestamp": "2020-04-16T09:41:23.989Z"
}

```

### Error Response

**Condition** : If a feedback already exists for the indicated prediction.

**Code** : `409 CONFLICT`

**Content example** :

```json
{
  "detail": "A feedback already exist fot the requested prediction"
}
```

## GET FEEDBACK

Return all the inserted feedback in chronological order starting from the last one stored. 

**URL** : `/api/feedback/`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Parameters

**Last** : If specified, returns only the last _n_ feedback inserted.

**Since** : If specified (following the format _%m/%d/%Y_), returns only the feedback inserted since that date.

### Success Response

**Condition** : If everything is OK a the requested feedback are returned to the user.

**Code** : `200 OK`

**Content example**

```json
[
  {
    "prediction": "5e902d7168e8ad2a707f604f",
    "correctness": false,
    "new_label": ">50K"
  }
] 

```

### Error Response

**Condition** : If no feedback has been found in the database.

**Code** : `404 NOT FOUND`

**Content example** :

```json
{
  "detail": "No feedback found"
}
```

## GET SINGLE FEEDBACK

Return a single inserted feedback, given the ID of the prediction to which it is referred to. 

**URL** : `/api/feedback/{prediction_ID}`

**Method** : `GET`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the ID passed as argument is valid and the feedback associated to it exists, the feedback will be returned to the user.

**Code** : `200 OK`

**Content example**

```json
{
  "prediction": "string",
  "correctness": true,
  "new_label": "optional string"
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

## MODIFY SINGLE FEEDBACK

Modify a single inserted feedback, given the ID of the prediction to which it is referred to. 

**URL** : `/api/feedback/{prediction_ID}`

**Method** : `PUT`

**Auth required** : None

**Permissions required** : None

### Success Response

**Condition** : If the ID passed as argument is valid and the feedback associated to it exists, the new feedback inserted will replace the latter.

**Code** : `200 OK`

**Content example**

```json
{
  "prediction": "5e902d7168e8ad2a707f604f",
  "correctness": true,
  "timestamp": "2020-04-16T13:10:00.989Z"
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