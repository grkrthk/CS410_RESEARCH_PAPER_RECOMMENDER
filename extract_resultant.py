resultant_path = "/home/grk/cs410_project/resultant.txt"
resfile = open(resultant_path, "a") 
path = "/home/grk/cs410_project/clustered_results/cluster"
for i in range(0, 9):
          unique_file_id_path = path + str(i) + "/" + "cluster_folder/" + "unique_ids.txt"
          topic_keywords_path = path + str(i) + "/" + "cluster_folder/" + "topic_keywords.txt"
          topicfile = open(topic_keywords_path, "r")
          keywords_buf = topicfile.readlines()
          keywords = ''.join(keywords_buf)
          uniqueidfile = open(unique_file_id_path, "r")        
          for line in uniqueidfile:
                              line = line.rstrip('\n')
                              line = line + ":" + keywords                              
                              print line
                              resfile.write(line + "\n")

