import xml.etree.ElementTree as ET
import pandas as pd
import os

#  *******************************START:Code to generate TSV file for Pair Master data ************************
def xmldata(fileName, xml_data, wantPairData):
    parsed_xml = ET.parse(fileName)
    root = parsed_xml.getroot()
    testCase = 'DDI2013'
    print(fileName)
    sentenceTag = root.findall('sentence')

    if wantPairData == True:
        dataFrameColumnsP = ['sentence', 'Pair_Id', 'E2', 'E1', 'DDI', 'type']
        xml_dataP = pd.DataFrame(columns=dataFrameColumnsP)
        for sentence in sentenceTag:
            pairTags = sentence.findall('pair')
            for tag in pairTags:
                xml_dataP = xml_dataP.append(pd.Series([sentence.attrib.get('text'), tag.attrib.get('id'),
                                                        tag.attrib.get('e1'), tag.attrib.get('e2'),
                                                        tag.attrib.get('ddi'), tag.attrib.get('type')],
                                                       index=dataFrameColumnsP), ignore_index=True)
    else:
        dataFrameColumnsE = ['sentence', 'E2', 'E1', 'DDI', 'type']
        xml_dataP = pd.DataFrame(columns=dataFrameColumnsE)

        for sentence in sentenceTag:
            pairTags = sentence.findall('entity')
            for tag in pairTags:
                xml_dataP = xml_dataP.append(pd.Series([sentence.attrib.get('text'),tag.attrib.get('e1'),
                                                        tag.attrib.get('e2'),tag.attrib.get('ddi'),
                                                        tag.attrib.get('type')],
                                                       index=dataFrameColumnsE), ignore_index=True)

    frames = [xml_data, xml_dataP]
    xml_data =pd.concat(frames)
    return  xml_data

Path = "D:/University/Sem2/Advance Linguistics/Project/Dataset/Train/combined"
# Path  = "D:/University/Sem1/Linguistics/Project/drugbank_all_full_database/semeval_task9_train_pair/Test/test"
filelist = os.listdir(Path)


xml_data = pd.DataFrame()

count = 0
for i in filelist:
    if i.endswith(".xml"):
        # if count < 1:
        xml_data = xmldata("Train/combined/" + i, xml_data, False) # for train
        # xml_data = xmldata("../Test/test/"+i, xml_data)
        count += 1

print(xml_data)
print("Files visited were: ", count)
xml_data.to_csv("DrugPair_MasterData_Train.tsv", sep="\t")

#  *******************************END:Code to generate TSV file for Pair Master data ************************