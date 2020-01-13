// export const ACTIVE_LEARNING_SERVER = "http://localhost:5000";
export const ACTIVE_LEARNING_SERVER = "http://73.254.202.129:5000";
export const COOKIE_LOGIN = "active_learning_ner_user_name_cookie";
export const PROGRESS_WAIT_TIME = 10000 * 3; // 50 seconds

// TURK Study
export const TURK_DATASET = 5;
export const TURK_CLASS = 'ADR';
export const TURK_CONDITION_SHOW_PREDICTIONS = true;

const configuration = {
    ACTIVE_LEARNING_SERVER,
    COOKIE_LOGIN,
    PROGRESS_WAIT_TIME,
}
export default configuration;


export const TURK_MAIN_PAGE_DATA = {
    example_id: 1,
    example: ["Muscle","aches","and","weakness","in","neck",",","arms",",","shoulders",",","upper","back",",","legs",".","Severe","pain","in","left","foot","and","third","and","fourth","toes",".","Pain","and","weakness","increased","until","I","needed","help","dressing","and","combing","my","hair",".","I","could","not","do","daily","activities","such","as","pick","up","a","cup","of","coffee","or","put","on","a","seatbelt","without","significant","pain",".","The","ball","of","my","left","foot","was","so","painful","I","could","not","put","pressure","on","it","when","walking","so","I","walked","on","the","side","of","my","foot",".","Symptoms","began","12","weeks","after","starting","Lipitor",".","I","reported","to","my","cardiologist","and","internist","who","both","denied","problems","were","related","to","Lipitor","because","my","CPK","was","only","mildly","elevated","(","250",")",".","I","also","saw","a","podiatrist",",","chiropractor",",","and","spine","specialist","with","no","relief",".","When","it","got","so","bad","I","could","not","take","it","anymore","and","did","not","care","about","my","cholesterol","I","took","myself","off","of","Lipitor","and","got","a","50","%","improvement","after","one","week",".","I","then","saw","a","neurologist","who","diagnosed","me","with","statin","induced","myalgia",".","He","said","he","sees","a","lot","of","patients","with","slightly","elevated","CPK","that","many","physicians","do","not","recognized","as","a","symptom","of","a","statin","intolerance",".","I","have","been","off","of","Lipitor","for","two","years","and","continue","to","improve","very","slowly",",","but","I","still","suffer","from","some","pain","and","weakness",".","I","can","now","walk","normal",",","but","my","toes","are","still","quite","painful","at","times","."],
    example_predictions: [[[0,2],[3,4],[17,18],[27,28],[29,30],[69,73],[186,187],[236,237],[238,239],[248,253]]],
};