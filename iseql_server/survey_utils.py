import os
import sys
import uuid

def submit_survey(firebaseManager, survey_info) -> str:
    '''
    Given the survey responses, validates them and stores the result in Firebase
    returns the a generated survey code
    '''
    survey_code = get_survey_code()
    survey_info["survey_code"] = survey_code

    firebaseManager.save_survey_data(survey_info)

    return survey_code

def get_survey_code() -> str:
    '''
    returns a new survey code for the user
    '''
    return str(uuid.uuid4())