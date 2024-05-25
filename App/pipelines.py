import pandas as pd
import pickle

def predict_loan_output(b_data):
    return pipeline_out(preprocess(b_data))

def pipeline_out(X):
    output = []
    class_pipeline = pickle.load(open("pipeline_class.pkl", 'rb'))
    output.append(class_pipeline.predict(X))

    reg_pipeline = pickle.load(open("pipeline_reg.pkl", 'rb'))
    output.append(reg_pipeline.predict(X))

    #print(process_data.keys())

    return output

# Define the base of our entry data for borrower in one dictionary
process_data = {}

def preprocess(b_data):
    '''
    INPUT: A list of a single list of results from a single borrower
    OUTPUT: A DataFrame with the X values of our pipelines
    '''    
    # Numeric data setup
    numeric_preprocess(b_data)

    # VerificationType
    vc_process(b_data)

    # LanguageCode
    lang_process(b_data)

    # Gender
    g_process(b_data)

    # Country
    c_process(b_data)

    # UseOfLoan
    uof_process(b_data)

    # Education
    edu_process(b_data)

    # Marital Status
    ms_process(b_data)

    # Employment Status
    es_process(b_data)

    # Employment Duration Current Employer
    ed_process(b_data)

    # Occupation Area
    oc_process(b_data)

    # Home Ownership Type
    hot_process(b_data)

    # Rating
    rating_process(b_data)

    # Credit Score Es MicroL
    cs_process(b_data)
    
    #print("Process Data Type: ", process_data)
    return pd.DataFrame([process_data.values()], columns=process_data.keys())



def numeric_preprocess(b_data):
    process_data['BidsPortfolioManager'] = int(b_data[0])
    process_data['BidsApi'] = int(b_data[1])
    process_data['BidsManual'] = int(b_data[2])
    process_data['NewCreditCustomer'] = int(b_data[3])
    process_data['Age'] = int(b_data[4])
    process_data['AppliedAmount'] = float(b_data[5])
    process_data['Interest'] = float(b_data[6])
    process_data['MonthlyPayment'] = float(b_data[7])
    process_data['IncomeTotal'] = float(b_data[8])
    process_data['ExistingLiabilities'] = int(b_data[9])
    process_data['LiabilitiesTotal'] = float(b_data[10])
    process_data['RefinanceLiabilities'] = int(b_data[11])
    process_data['DebtToIncome'] = float(b_data[12])
    process_data['FreeCash'] = float(b_data[13]) #(Calculate),
    process_data['Restructured'] = int(b_data[14])
    process_data['PrincipalPaymentsMade'] = float(b_data[15])
    process_data['InterestAndPenaltyPaymentsMade'] = float(b_data[16])
    process_data['PrincipalBalance'] = float(b_data[17])
    process_data['InterestAndPenaltyBalance'] = float(b_data[18])
    process_data['PreviousRepaymentsBeforeLoan'] = float(b_data[19])
    process_data['PreviousEarlyRepaymentsCountBeforeLoan'] = int(b_data[20])
    process_data['Ava_Inc'] = float(b_data[8]) - float(b_data[10]) #(Calculate),
    process_data['InterestAmount'] = float(b_data[5]) * float(b_data[6])/100 #(Calculate)]


def vc_process(b_data):   
    process_data['VerificationType_Income unverified'] = 0
    process_data['VerificationType_Income unverified,cross-referenced by phone'] = 0
    process_data['VerificationType_Income verified'] = 0
    process_data['VerificationType_Not Set'] = 0

    if b_data[21] == 'Income unverified':
        process_data['VerificationType_Income unverified'] = 1

    elif b_data[21] == 'Income unverified,cross-referenced by phone':
        process_data['VerificationType_Income unverified,cross-referenced by phone'] = 1

    elif b_data[21] == "'Income verified'":
        process_data['VerificationType_Income verified'] = 1

    elif (b_data[21] == "None"):
        process_data['VerificationType_Not Set'] = 1
    
    
def lang_process(b_data):
    # LanguageCode
    process_data['LanguageCode_Estonian'] = 0
    process_data['LanguageCode_Finnish'] = 0
    process_data['LanguageCode_German'] = 0
    process_data['LanguageCode_Other'] = 0
    process_data['LanguageCode_Russian'] = 0
    process_data['LanguageCode_Slovakian'] = 0
    process_data['LanguageCode_Spanish'] = 0

    if b_data[22] == 'Estonian':
        process_data['LanguageCode_Estonian'] = 1
        
    elif b_data[22] == 'Finnish':
        process_data['LanguageCode_Finnish'] = 1
        
    elif b_data[22] == 'German':
        process_data['LanguageCode_German'] = 1
        
    elif b_data[22] == 'Other':
        process_data['LanguageCode_Other'] = 1
    
    elif b_data[22] == 'Russian':
        process_data['LanguageCode_Russian'] = 1
    
    elif b_data[22] == 'Slovakian':
        process_data['LanguageCode_Slovakian'] = 1

    elif b_data[22] == 'Spanish':
        process_data['LanguageCode_Spanish'] = 1

    
    
def g_process(b_data):
    # Gender
    process_data['Gender_Male'] = 0
    process_data['Gender_Undefined'] = 0

    if b_data[23] == 'Male':
        process_data['Gender_Male'] = 1
    
    elif b_data[23] == 'None':
        process_data['Gender_Undefined'] = 1
    
    
def c_process(b_data):
    # Country
    process_data['Country_ES'] = 0
    process_data['Country_FI'] = 0
    process_data['Country_SK'] = 0

    if b_data[24] == 'ES':
        process_data['Country_ES'] = 1
    elif b_data[24] == 'FI':
        process_data['Country_FI'] = 1
    elif b_data[24] == 'SK':
        process_data['Country_SK'] = 1
    

def uof_process(b_data):
    # UseOfLoan
    process_data['UseOfLoan_Acquisition of means of transport'] = 0
    process_data['UseOfLoan_Acquisition of real estate'] = 0
    process_data['UseOfLoan_Acquisition of stocks'] = 0
    process_data['UseOfLoan_Business'] = 0
    process_data['UseOfLoan_Education'] = 0
    process_data['UseOfLoan_Health'] = 0
    process_data['UseOfLoan_Home improvement'] = 0
    process_data['UseOfLoan_Loan consolidation'] = 0
    process_data['UseOfLoan_Not Set'] = 0
    process_data['UseOfLoan_Other'] = 0
    process_data['UseOfLoan_Other business'] = 0
    process_data['UseOfLoan_Purchase of machinery equipment'] = 0
    process_data['UseOfLoan_Real estate'] = 0
    process_data['UseOfLoan_Travel'] = 0
    process_data['UseOfLoan_Vehicle'] = 0
    process_data['UseOfLoan_Working capital financing'] = 0
    
    if b_data[25] == 'Acquisition of means of transport':
        process_data['UseOfLoan_Acquisition of means of transport'] = 1
        
    elif b_data[25] == 'Acquisition of real estate':
        process_data['UseOfLoan_Acquisition of real estate'] = 1
        
    elif b_data[25] == 'Acquisition of stocks':
        process_data['UseOfLoan_Acquisition of stocks'] = 1
        
    elif b_data[25] == 'Business':
        process_data['UseOfLoan_Business'] = 1
        
    elif b_data[25] == 'Education':
        process_data['UseOfLoan_Education'] = 1
        
    elif b_data[25] == 'Health':
        process_data['UseOfLoan_Health'] = 1
        
    elif b_data[25] == 'Home improvement':
        process_data['UseOfLoan_Home improvement'] = 1
        
    elif b_data[25] ==  'Loan consolidation':
        process_data['UseOfLoan_Loan consolidation'] = 1
        
    elif b_data[25] == 'Other':
        process_data['UseOfLoan_Other'] = 1
        
    elif b_data[25] == 'Other business':
        process_data['UseOfLoan_Other business'] = 1
        
    elif b_data[25] == 'Purchase of machinery equipment':
        process_data['UseOfLoan_Purchase of machinery equipment'] = 1
        
    elif b_data[25] == 'Real estate':
        process_data['UseOfLoan_Real estate'] = 1
        
    elif b_data[25] == 'Travel':
        process_data['UseOfLoan_Travel'] = 1
        
    elif b_data[25] == 'Vehicle':
        process_data['UseOfLoan_Vehicle'] = 1
        
    elif b_data[25] == 'Working capital financing':
        process_data['UseOfLoan_Working capital financing'] = 1
    

    
def edu_process(b_data):
    # Education
    process_data['Education_Higher education'] = 0
    process_data['Education_Not Present'] = 0
    process_data['Education_Primary education'] = 0
    process_data['Education_Secondary education'] = 0
    process_data['Education_Vocational education'] = 0
    

    if b_data[26] == 'Primary education':
        process_data['Education_Primary education'] = 1
        
    elif b_data[26] == 'Vocational education':
        process_data['Education_Vocational education'] = 1
    
    elif b_data[26] == 'Secondary education':
        process_data['Education_Secondary education'] = 1

    elif b_data[26] == 'Higher education':
        process_data['Education_Higher education'] = 1     
    
    
def ms_process(b_data):
    # Marital Status
    process_data['MaritalStatus_Divorced'] = 0
    process_data['MaritalStatus_Married'] = 0
    process_data['MaritalStatus_Not Specified'] = 0
    process_data['MaritalStatus_Single'] = 0
    process_data['MaritalStatus_Widow'] = 0
    
    if b_data[27] == 'Divorced':
        process_data['MaritalStatus_Divorced'] = 1
        
    elif b_data[27] == 'Married':
        process_data['MaritalStatus_Married'] = 1
        
    elif b_data[27] == 'Single':
        process_data['MaritalStatus_Single'] = 1
        
    elif b_data[27] == "Widow":
        process_data['MaritalStatus_Widow'] = 1
    
    elif b_data[27] == "None":
        process_data['MaritalStatus_Not Specified'] = 1
        
    
def es_process(b_data):
    # Employment Status
    process_data['EmploymentStatus_Fully employed'] = 0
    process_data['EmploymentStatus_Not present'] = 0
    process_data['EmploymentStatus_Partially employed'] = 0
    process_data['EmploymentStatus_Retiree'] = 0
    process_data['EmploymentStatus_Self-employed'] = 0

    if b_data[28] == 'Fully employed':
        process_data['EmploymentStatus_Fully employed'] = 1
    elif b_data[28] == 'None':
        process_data['EmploymentStatus_Not present'] = 1
    elif b_data[28] == 'Partially employed':    
        process_data['EmploymentStatus_Partially employed'] = 1
    elif b_data[28] == 'Retiree':
        process_data['EmploymentStatus_Retiree'] = 1
    elif b_data[28] == 'Self-employed':
        process_data['EmploymentStatus_Self-employed'] = 1

    
    
def ed_process(b_data):
    # Employment Duration Current Employer
    process_data['EmploymentDurationCurrentEmployer_Other'] = 0
    process_data['EmploymentDurationCurrentEmployer_Retiree'] = 0
    process_data['EmploymentDurationCurrentEmployer_TrialPeriod'] = 0
    process_data['EmploymentDurationCurrentEmployer_UpTo1Year'] = 0
    process_data['EmploymentDurationCurrentEmployer_UpTo2Years'] = 0
    process_data['EmploymentDurationCurrentEmployer_UpTo3Years'] = 0
    process_data['EmploymentDurationCurrentEmployer_UpTo4Years'] = 0
    process_data['EmploymentDurationCurrentEmployer_UpTo5Years'] = 0
    
    if b_data[29] == "None":
        process_data['EmploymentDurationCurrentEmployer_Other'] = 1
    elif b_data[29] == "Retiree":
        process_data['EmploymentDurationCurrentEmployer_Retiree'] = 1
    elif b_data[29] == "TrialPeriod":
        process_data['EmploymentDurationCurrentEmployer_TrialPeriod'] = 1
    elif b_data[29] == "UpTo1Year":
        process_data['EmploymentDurationCurrentEmployer_UpTo1Year'] = 1
    elif b_data[29] == "UpTo2Years":
        process_data['EmploymentDurationCurrentEmployer_UpTo2Years'] = 1
    elif b_data[29] == "UpTo3Years":
        process_data['EmploymentDurationCurrentEmployer_UpTo3Years'] = 1
    elif b_data[29] == "UpTo4Years":
        process_data['EmploymentDurationCurrentEmployer_UpTo4Years'] = 1
    elif b_data[29] == "UpTo5Years":
        process_data['EmploymentDurationCurrentEmployer_UpTo5Years'] = 1

    
def oc_process(b_data):
    # Occupation Area
    process_data['OccupationArea_Agriculture, forestry and fishing'] = 0
    process_data['OccupationArea_Art and entertainment'] = 0
    process_data['OccupationArea_Civil service & military'] = 0
    process_data['OccupationArea_Construction'] = 0
    process_data['OccupationArea_Education'] = 0
    process_data['OccupationArea_Energy'] = 0
    process_data['OccupationArea_Finance and insurance'] = 0
    process_data['OccupationArea_Healthcare and social help'] = 0
    process_data['OccupationArea_Hospitality and catering'] = 0
    process_data['OccupationArea_Info and telecom'] = 0
    process_data['OccupationArea_Mining'] = 0
    process_data['OccupationArea_Not present'] = 0
    process_data['OccupationArea_Other'] = 0
    process_data['OccupationArea_Processing'] = 0
    process_data['OccupationArea_Real-estate'] = 0
    process_data['OccupationArea_Research'] = 0
    process_data['OccupationArea_Retail and wholesale'] = 0
    process_data['OccupationArea_Transport and warehousing'] = 0
    process_data['OccupationArea_Utilities'] = 0
    
    if b_data[30] == 'Agriculture, forestry and fishing':
        process_data['OccupationArea_Agriculture, forestry and fishing'] = 1
    elif b_data[30] == 'Art and entertainment':
        process_data['OccupationArea_Art and entertainment'] = 1
    elif b_data[30] == "Civil service & military":
        process_data['OccupationArea_Civil service & military'] = 1
    elif b_data[30] == "Construction":
        process_data['OccupationArea_Construction'] = 1
    elif b_data[30] == "Education":
        process_data['OccupationArea_Education'] = 1
    elif b_data[30] == 'Energy':
        process_data['OccupationArea_Energy'] = 1
    elif b_data[30] == 'Finance and insurance':
        process_data['OccupationArea_Finance and insurance'] = 1
    elif b_data[30] == 'Healthcare and social help':
        process_data['OccupationArea_Healthcare and social help'] = 1
    elif b_data[30] == 'Hospitality and catering':
        process_data['OccupationArea_Hospitality and catering'] = 1
    elif b_data[30] == 'Info and telecom':
        process_data['OccupationArea_Info and telecom'] = 1
    elif b_data[30] == 'Mining':
        process_data['OccupationArea_Mining'] = 1
    elif b_data[30] == 'None':
        process_data['OccupationArea_Other'] = 1
    elif b_data[30] == 'Processing':
        process_data['OccupationArea_Processing'] = 1
    elif b_data[30] == 'Real-estate':
        process_data['OccupationArea_Real-estate'] = 1
    elif b_data[30] == 'Research':
        process_data['OccupationArea_Research'] = 1
    elif b_data[30] == 'Retail and wholesale':
        process_data['OccupationArea_Retail and wholesale'] = 1
    elif b_data[30] == 'Transport and warehousing':
        process_data['OccupationArea_Transport and warehousing'] = 1
    elif b_data[30] == 'Utilities':
        process_data['OccupationArea_Utilities'] = 1
            
    
def hot_process(b_data):
    # Home Ownership Type
    process_data['HomeOwnershipType_Homeless'] = 0
    process_data['HomeOwnershipType_Joint ownership'] = 0
    process_data['HomeOwnershipType_Joint tenant'] = 0
    process_data['HomeOwnershipType_Living with parents'] = 0
    process_data['HomeOwnershipType_Mortgage'] = 0
    process_data['HomeOwnershipType_Not specified'] = 0
    process_data['HomeOwnershipType_Other'] = 0
    process_data['HomeOwnershipType_Owner'] = 0
    process_data['HomeOwnershipType_Owner with encumbrance'] = 0
    process_data['HomeOwnershipType_Tenant,pre-furnished property'] = 0

    if b_data[31] == 'Homeless':
        process_data['HomeOwnershipType_Homeless'] = 1
    elif b_data[31] == 'Joint ownership':
        process_data['HomeOwnershipType_Joint ownership'] = 1
    elif b_data[31] == 'Joint tenant':
        process_data['HomeOwnershipType_Joint tenant'] = 1
    elif b_data[31] == 'Living with parents':
        process_data['HomeOwnershipType_Living with parents'] = 1
    elif b_data[31] == 'Mortgage':
        process_data['HomeOwnershipType_Mortgage'] = 1
    elif b_data[31] == 'None':
        process_data['HomeOwnershipType_Other'] = 1
    elif b_data[31] == 'Owner':
        process_data['HomeOwnershipType_Owner'] = 1
    elif b_data[31] == 'Owner with encumbrance':
        process_data['HomeOwnershipType_Owner with encumbrance'] = 1
    elif b_data[31] == 'Tenant,pre-furnished property':
        process_data['HomeOwnershipType_Tenant,pre-furnished property'] = 1

    
    
def rating_process(b_data):
    # Rating
    process_data['Rating_AA'] = 0
    process_data['Rating_B'] = 0
    process_data['Rating_C'] = 0
    process_data['Rating_D'] = 0
    process_data['Rating_E'] = 0
    process_data['Rating_F'] = 0
    process_data['Rating_HR'] = 0

    if b_data[32] == 'AA':
        process_data['Rating_AA'] = 1
    elif b_data[32] == 'B':
        process_data['Rating_B'] = 1
    elif b_data[32] == 'C':
        process_data['Rating_C'] = 1
    elif b_data[32] == 'D':
        process_data['Rating_D'] = 1
    elif b_data[32] == 'E':
        process_data['Rating_E'] = 1
    elif b_data[32] == 'F':
        process_data['Rating_F'] = 1
    elif b_data[32] == 'HR':
        process_data['Rating_HR'] = 1

    
    
def cs_process(b_data):
    # Credit Score Es MicroL
    process_data['CreditScoreEsMicroL_M1'] = 0
    process_data['CreditScoreEsMicroL_M10'] = 0
    process_data['CreditScoreEsMicroL_M2'] = 0
    process_data['CreditScoreEsMicroL_M3'] = 0
    process_data['CreditScoreEsMicroL_M4'] = 0
    process_data['CreditScoreEsMicroL_M5'] = 0
    process_data['CreditScoreEsMicroL_M6'] = 0
    process_data['CreditScoreEsMicroL_M7'] = 0
    process_data['CreditScoreEsMicroL_M8'] = 0
    process_data['CreditScoreEsMicroL_M9'] = 0
    
    if b_data[33] == 'M1':
        process_data['CreditScoreEsMicroL_M1'] = 1
    elif b_data[33] == 'M10':
        process_data['CreditScoreEsMicroL_M10'] = 1
    elif b_data[33] == 'M2':
        process_data['CreditScoreEsMicroL_M2'] = 1
    elif b_data[33] == 'M3':
        process_data['CreditScoreEsMicroL_M3'] = 1
    elif b_data[33] == 'M4':
        process_data['CreditScoreEsMicroL_M4'] = 1
    elif b_data[33] == 'M5':
        process_data['CreditScoreEsMicroL_M5'] = 1
    elif b_data[33] == 'M6':
        process_data['CreditScoreEsMicroL_M6'] = 1
    elif b_data[33] == 'M7':
        process_data['CreditScoreEsMicroL_M7'] = 1
    elif b_data[33] == 'M8':
        process_data['CreditScoreEsMicroL_M8'] = 1
    elif b_data[33] == 'M9':
        process_data['CreditScoreEsMicroL_M9'] = 1
    