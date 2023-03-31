# In[]:
import os
import json
import subprocess

# In[ ]:
# ** loading config **
with open('./{}'.format("config_scripts.json")) as file:
    scripts = json.load(file)

# In[ ]:
def question_yes_or_no(question):
    if scripts['skip_yes_no_questions']:
        return True
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False
        
# In[]:
def main():
    print('main executed')
    for script in scripts["scripts"]:
        scriptpath = script["path"]
        
        if script["skip_run"]:
            continue
        
        if question_yes_or_no("Execute the '{}' script".format(scriptpath)):
            cwd = os.getcwd()
            filefullpath = os.path.abspath(scriptpath)
            os.chdir(os.path.dirname(filefullpath))
            command = 'python {}'.format(filefullpath)
            print(command)
            retcode = subprocess.call(command, shell=True)
            os.chdir(cwd)
            print("retcode : {}".format(retcode))

# In[]:
if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
    