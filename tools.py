#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: tools.py
Author: lifayuan(lifayuan@baidu.com)
Date: 2019/02/25 15:54:09
"""
import os
import sys
import json
import re

import numpy as np
import jieba

def check_path(path):
    """

    :param path:
    :return:
    """
    path = os.path.abspath(path)
    path_dirname = os.path.dirname(path)
    if not os.path.exists(path_dirname):
        os.makedirs(path_dirname)


def modify_datasets(path):
    """
        brief info for: modify_datasets
        
        Args:
            path :
        Return:   
        Raise: 
    """
    out = open(path + ".new", "w")
    with open(path) as fin:
        datasets = json.loads(fin.read())
        # print json.dumps(datasets[0], ensure_ascii=False).encode("utf8")
        new_datasets = []
        for data in datasets:
            for key in ["home_line", "vis_line"]:
                if "continuous_result" in data[key]:
                    team_status = data[key]["continuous_result"]
                    del data[key]["continuous_result"]
                    if u"连胜" in team_status:
                        data[key]["TEAM-STATUS_WIN"] = team_status.rstrip(u"连胜")
                        data[key]["TEAM-STATUS_LOSS"] = "0"
                    else:
                        data[key]["TEAM-STATUS_LOSS"] = team_status.rstrip(u"连负")
                        data[key]["TEAM-STATUS_WIN"] = "0"
            new_datasets.append(data)
        out.write(json.dumps(new_datasets, ensure_ascii=False).encode("utf8"))
        out.close()


def modify_summary(path):
    """
        brief info for: modify_summay
        
        Args:
            path :
        Return:   
        Raise: 
    """
    pattern = re.compile("\s([0-9]{2,3})\s-\s([0-9]{2,3})\s(\S+)")

    with open(path) as fin:
        for line in fin:
            text = line.strip()
            sents = text
            m = pattern.search(text)
            if m:
                s1 = int(m.group(1))
                s2 = int(m.group(2))
                term = m.group(3)
                print(s1, s2, term)
                '''
                if s1 < s2 and term != "）":
                    print s1, s2, term

                '''


def modify_summary_v2(path):
    """
        brief info for: modify_summary_v2
        
        Args:
            path :
        Return:   
        Raise: 
    """
    pattern = re.compile("\s([0-9]{2,3})\s-\s([0-9]{2,3})\s(\S+)")
    action_dict = {u"击败": u"不敌", u"大胜": u"惨败给", u"战胜": u"不敌", \
            u"痛宰": u"惨败给", u"轻取": u"败给", u"险胜": u"惜败"}
    output = open(path + ".new", "w")
    with open(path) as fin:
        data = json.loads(fin.read())
    new_data = []
    for match in data:
        new_data.append(match)
        summary = " ".join(match["summary"])
        home_name = match["home_name"]
        home_city = match["home_city"]

        vis_name = match["vis_name"]
        vis_city = match["vis_city"]
        trans_flag = 0
        # print "before:", summary
        sents = segment(summary)
        new_text = ""
        for sent in sents:
            m = pattern.search(sent)
            if m:
                s1 = m.group(1)
                s2 = m.group(2)
                term = m.group(3)
                # print s1, s2, term
                # print term
                if term not in action_dict:
                    new_text += sent
                    continue

                sent = sent.replace(" ", "")
                if home_name in sent and vis_name in sent:
                    sent = sent.replace(home_city, "").\
                            replace(vis_city, "")
                    '''
                    print('nnnnn')
                    print(sent)
                    '''
                    home_part = get_replace_part(sent, home_name)
                    vis_part = get_replace_part(sent, vis_name)
                    '''
                    print home_part
                    print vis_part
                    '''
                    sent = transform(sent, home_part, vis_part, s1, s2,\
                            term, action_dict)
                    trans_flag = 1
                sent = " ".join([" "] + list(jieba.cut(sent)))

                new_text += sent

            else:
                new_text += sent 
        print("after:", new_text)
        match["summary"] = new_text.split(" ")
        if trans_flag:
            new_data.append(match)
    new_data_idx_list = np.arange(len(new_data))
    np.random.shuffle(new_data_idx_list)
    new_data = [new_data[idx] for idx in list(new_data_idx_list)]
    output.write(json.dumps(new_data, ensure_ascii=False).encode("utf8")) 

def transform(sent, home_part, vis_part, s1, s2, term, action_dict):
    """
        brief info for: transform
        
        Args:
            sent :
            home_part :
            vis_part :
            s1 :
            s2 :
            term :
        Return:   
        Raise: 
    """
    sent = sent.replace(home_part, "<home_part>")
    sent = sent.replace(vis_part, "<vis_part>")
    sent = sent.replace("<home_part>", vis_part)
    sent = sent.replace("<vis_part>", home_part)
    
    sent = sent.replace(s1, "<s1>")
    sent = sent.replace(s2, "<s2>")
    sent = sent.replace("<s1>", s2)
    sent = sent.replace("<s2>", s1)
    
    '''
    if u"客场" not in sent and u"主场" not in sent:
        print "wwwwwww:", sent
    '''

    if u"客场" in sent:
        sent = sent.replace(u"客场", u"主场")
    elif u"主场" in sent:
        sent = sent.replace(u"主场", u"客场")
    elif u"做客" in sent:
        sent = sent.replace(u"做客", u"主场")

    sent = sent.replace(term, action_dict[term])
    return sent



def get_replace_part(sent, name):
    """
        brief info for: get_replace_part
        
        Args:
            sent :
            name :
        Return:   
        Raise: 
    """
    p_list = []
    p_list.append(re.compile(name + u"队（\d+胜\d+负）"))
    p_list.append(re.compile(name + u"（\d+胜\d+负）"))
    for p in p_list:
        m = p.search(sent)
        if m:
            part = m.group()
            break
    if not m:
        part = name + u"队"
        if part in sent:
            return part
        else:
            return name
    return part


def segment(text):
    """

    :param text:
    :return:
    """
    delim_list = [u"，", u"。", u"；"]
    sentences = []
    sent = ""
    for char in text:
        if char in delim_list:
            sent += char
            sentences.append(sent)
            sent = ""
        else:
            sent += char
    '''
    for sent in sentences:
        print sent
    '''
    return sentences


def output_team_from_summary(path):
    """

    :param path:
    :return:
    """
    with open("city_name") as fin:
        teams = [line.strip().split("\t")[0] for line in fin]

    with open(path) as fin:
        for line in fin:
            text = line.strip().replace(" ", "")
            out = []
            for team in teams:
                if team in text:
                    out.append(team)
            # out = sorted(list(set(out)))
            out = list(set(out))
            print(" ".join(out))

def fill_empty_value(path):
    """

    :param path:
    :return:
    """
    with open(path) as fin:
        trdata = json.loads(fin.read())
    trdata_tmp = []
    for item in trdata:
        if len(item["summary"]) <= 10:
            continue
        '''
        for k,v in item["box_score"]["PLAYER_NAME"].items():
            item["box_score"]["PLAYER_NAME"][k] = v.replace("-", " ")
        '''
        for op in item["box_score"]:
            for k,v in item["box_score"][op].items():
                if v == "":
                    item["box_score"][op][k] = "N/A"
        
        if item["home_line"].get("TEAM-STATUS_LOSS", "") == "":
            item["home_line"]["TEAM-STATUS_LOSS"] = "N/A"
        if item["home_line"].get("TEAM-STATUS_WIN", "") == "":
            item["home_line"]["TEAM-STATUS_WIN"] = "N/A"
        if item["vis_line"].get("TEAM-STATUS_LOSS", "") == "":
            item["vis_line"]["TEAM-STATUS_LOSS"] = "N/A"
        if item["vis_line"].get("TEAM-STATUS_WIN", "") == "":
            item["vis_line"]["TEAM-STATUS_WIN"] = "N/A"

            
        trdata_tmp.append(item)
    trdata = trdata_tmp
    with open(path + ".fill", "w") as fout:
        fout.write(json.dumps(trdata, ensure_ascii=False).encode("utf8"))


def get_player_team(path):
    """

    :param path:
    :return:
    """
    with open(path) as fin:
        data_list = json.loads(fin.read())
    out = []
    for data in data_list:
        for idx in data["box_score"]["PLAYER_NAME"]:
            name = data["box_score"]["PLAYER_NAME"][idx]
            city = data["box_score"]["TEAM_CITY"][idx]
            team = ""
            if city == data["home_city"]:
                team = data["home_name"]
            elif city == data["vis_city"]:
                team = data["vis_name"]
            name = name.replace(" ", "-")
            out.append((name, team))
    out = list(set(out))
    for name, team in out:
        print("%s\t%s" % (name, team))
                

def get_target_vocab(path_list):
    """

    :param path_list:
    :return:
    """
    vocab = set()
    for path in path_list:
        for line in open(path):
            tokens = line.strip().split()
            vocab.update(tokens)
    for token in vocab:
        print(token)


def get_quarter_tuple(text):
    """

    :param text:
    :return:
    """
    data = text.split()
    num = len(data)
    size = 5
    sample = {}
    for i in range(num/size):
        for j in range(size):
            idx = i * size + j
            if j not in sample:
                sample[j] = [data[idx]]
            else:
                sample[j].append(data[idx])
    
    for k in sample:
        value_num = len(sample[k])
        if value_num % 4 != 0:
            limit = value_num / 4 + 1
        else:
            limit = value_num / 4
        
        records = []
        for i in range(limit):
            start_idx = i * 4
            end_idx = (i + 1) * 4 
            item = u"￨".join(sample[k][start_idx: end_idx])
            records.append(item) 
        sample[k] = records

    for k in sample:
        print(" ".join(sample[k]))


if __name__ == "__main__":
    # modify_datasets("./realdata/test.json")
    # modify_datasets("./realdata2/valid.json")
    # modify_datasets("./nba_data/test.json")
    # modify_summary("./realdata/train.summary")
    # modify_summary("crawler/data/match_boxscore_data.summary")
    pass

