#%% IMPORT MODULES

import pandas as pd
import numpy as np
import random
import sys
from enum import Enum

#%% PARSER ACTION ENUMERATION

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    SHIFT = 2

#%% RANDOM ORACLE, FOR TESTING

def random_oracle(stack, input_buffer):
    # Pick a random action.
    action = random.choice(list(Action))
    
    # Don't allow shift on empty buffer.
    if len(input_buffer.index) == 0:
        if len(stack) == 2:
            action = Action.RIGHT
        else:
            action = random.choice([Action.LEFT, Action.RIGHT])
        
    
    # Don't allow Action.LEFT on "root".
    if len(stack) == 2 and action == Action.LEFT and len(input_buffer.index) > 0:
        action = random.choice([Action.SHIFT, Action.RIGHT])
    
    return action
   
#%% RULE-BASED PARSER

def rule_based_oracle(stack, input_buffer, print_details=False):
    # Build some useful sets of POS that can be considered similar.
    NOMINAL = {"NOUN", "PRON", "PROPN"}
    BEFORENOUN = {"ADJ", "ADV", "ADP", "DET"}   
    
    # Extract the features:
    # POS of top and second elements on the stack.
    s1 = stack[-1].UPOS
    s2 = stack[-2].UPOS if len(stack) > 1 else None
    # POS of the first and second elements on the input buffer.
    b1 = input_buffer.iloc[0].UPOS if len(input_buffer.index) > 0 else None
    b2 = input_buffer.iloc[1].UPOS if len(input_buffer.index) > 1 else None 
    
    
    def __shift_on_nominal__():
        # Shift if we have a verb followed by a determiner, or an adverb, and then by a nominal.
        if b1 in NOMINAL:       
            if print_details: print("\nRULE_N1")
            return Action.SHIFT
        # Same, but account for "det" + "noun" instead of just "noun".
        if b1 in BEFORENOUN and b2 in NOMINAL:
            if print_details: print("\nRULE_N2")
            return Action.SHIFT 
    
    # Pick a random action, make sure we always do something
    action = random.choice(list(Action))
    
    # If the stack has size == 1, turn any action into a SHIFT.
    # NOTE: checking if the length of the stack or of the buffer is <= 2
    #       is still equivalent to using second-level features!
    #       It's the same as checking if the POS of some word is "none", 
    #       as the word doesn't exist.
    if len(stack) == 1:
        if print_details: print("\nRULE_1")
        return Action.SHIFT
    
    # Recoginze the pattern "verb" + "det" + "adv/adj" + "adv/adj". 
    # This pattern is usually followed by a nominal, so we have to shift.
    # This rule must be at the top as it's more specific than the others.
    if s2 == "VERB" and s1 == "DET":
        if b1 in ["ADV", "ADJ"] and b2 in ["ADV", "ADJ"]:
            if print_details: print("\nRULE_2")
            return Action.SHIFT
    
    # Match the pattern "adv" + "adj":
    if s2 == "ADV" and s1 == "ADJ":
        if print_details: print("\nRULE_3")
        return Action.LEFT   
    
    if s2 == "VERB" and s1 in BEFORENOUN:
        if print_details: print("\nRULE_4")
        temp_action = __shift_on_nominal__()
        if temp_action != None:
            return temp_action          
            
        # If we have a verb followed by an adposition/determiner, 
        # and another two adpositions/determiners/adjectives/adverbs, 
        # we can suppose that the first adp/det is an object of the verb.
        if b1 in BEFORENOUN and\
            b2 in BEFORENOUN:              
            if print_details: print("\nRULE_5")
            return Action.RIGHT
        
    # Shift if we have two det/adp/etc... before a noun (or something + noun),
    # as they are probably dependent on that noun.
    if s2 in BEFORENOUN and s1 in BEFORENOUN:
        temp_action = __shift_on_nominal__()
        if print_details: print("\nRULE_6")
        if temp_action != None:
            return temp_action     
    
    
    # If a verb is followed by a nominal, and there are no verbs after it,
    # (which could signify that the nominal is the subject/object of a subordinate clause),
    # we can say that the nominal depends on the verb before it.
    if s2 == "VERB" and s1 in NOMINAL:
        if b1 != "VERB":
            if print_details: print("\nRULE_7")
            return Action.RIGHT
        # If the buffer is empty, associate the noun to the verb.
        if len(input_buffer.index) == 0:
            if print_details: print("\nRULE_8")
            return Action.RIGHT
    
    # Match the pattern "pronoun" + "verb", "noun" + "verb", "proper noun" + "verb", e.g. "I study", etc...
    if s2 in NOMINAL and s1 == "VERB":
        if print_details: print("\nRULE_9")
        return Action.LEFT    
    
    # Match the pattern "det" + "noun" and "adposition" + "noun", e.g. "the apple", "to (the) teacher"
    if s2 in ["DET", "ADP"] and s1 == "NOUN":
        if print_details: print("\nRULE_10")
        return Action.LEFT
    
    # Match the pattern "adj" + "noun":
    if s2 == "ADJ" and s1 == "NOUN":
        if print_details: print("\nRULE_11")
        return Action.LEFT   
    

    # If we have "__ROOT__" and a verb on the stack, mark the verb as root, with a right arc.
    # Note: we leave this action at the very end, so that the main verb isn't removed too soon!
    # If we match the pattern too soon, return a shift!
    if s2 == "__ROOT__":
        if print_details: print("\nRULE_12")
        if s1 == "VERB":
            return Action.RIGHT if len(input_buffer.index) == 0 else Action.SHIFT
        else:
            return Action.SHIFT
        
    # Don't allow shift on empty buffer.
    if len(input_buffer.index) == 0:
        action = random.choice([Action.LEFT, Action.RIGHT])
    
    if print_details: print("\nRULE_RANDOM")
    return action
    
    


#%% DEPENDENCY PARSER

def dep_parser(input_line, oracle=random_oracle, print_details=False, print_to_file=None):
    
    # Create a special line for the "root" element of the parser.
    root = pd.Series(index=input_line.columns)
    root.loc["ID"] = 0
    root.loc["Form"] = "__ROOT__"
    root.loc["Lemma"] = "__ROOT__"
    root.loc["UPOS"] = "__ROOT__"
    
    # The stack is managed as an array, where each element is a pandas.Series,
    # and refers to a row of the input_line DataFrame.
    # Initially, only the root is contained.
    stack = [root]
    # For the buffer, we can keep track of the current position.
    buffer_index = 0
    
    # Count the steps.
    step_count = 0
    
    # Set the separator and the endline separator,
    # useful for printing pretty stuff and latex tables.
    sep = " & "
    endl_sep = "\\\\ \n"
    
    print("\n\n#################\n# START PARSING #\n#################\n\n")
    print("STEP" + sep + "STACK" + sep + "BUFFER" + sep + "ACTION" +  sep + "OUTPUT" + endl_sep, end="",\
          file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
    
    # Stop the parsing if the stack contains only "root" and the input buffer is empty.
    # The "root" is never removed from the stack, so the length is <= 1 only at the start and at the end.
    # When the input buffer is empty, it means that the buffer index is at the end of the DataFrame.
    while len(stack) > 1 or buffer_index < len(input_line.index):
        # Print the stack and the buffer.
        print(step_count, end=sep,\
          file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))     
        print([word.Form for word in stack], end=sep,\
          file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
        print(list(input_line.iloc[buffer_index:, 1]), end=sep,\
          file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
        
        # Features, to be written to a file.
        # POS of top and second elements on the stack.
        s1 = stack[-1].UPOS
        s2 = stack[-2].UPOS if len(stack) > 1 else "none"
        # POS of the first and second elements on the input buffer.
        b1 = input_line.iloc[buffer_index:, :].iloc[0].UPOS if len(input_line.iloc[buffer_index:, :].index) > 0 else "none"
        b2 = input_line.iloc[buffer_index:, :].iloc[1].UPOS if len(input_line.iloc[buffer_index:, :].index) > 1 else "none" 
        
        features = [s1, s2, b1, b2]
        
        # Call the oracle to find the most suitable action.
        # We pass the entire input buffer and stack to it, 
        # so the oracle implementation is fully transparent from the parser.
        action = oracle(stack, input_line.iloc[buffer_index:, :], print_details=print_details)
                
        # Check if the action is valid:
        if action not in Action:
            raise ValueError("UNKNOWN ACTION:", action)
            
        # SAFETY CHECKS:
        # This rules hold true regardless of the oracle that is used,
        # and make sure that we don't perform illegal actions.
            
        # Don't allow shift on empty buffer.
        if buffer_index >= len(input_line.index) and len(stack) == 2 and action == Action.SHIFT:
            if print_details: print("DONT SHIFT ON EMPTY BUFFER")
            action = Action.RIGHT
    
        # Don't allow Action.LEFT on "root".
        if len(stack) == 2 and action == Action.LEFT:
            raise ValueError("LEFT ACTION PERFORMED ON ROOT")
            
        # If the stack has size == 1, turn any action into a SHIFT.
        if len(stack) == 1 and action != Action.SHIFT:
            if print_details: print("FORCE SHIFT ON EMPTY STACK!")
            action = Action.SHIFT
            
        # Print the action.
        print(action, end=sep,\
          file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
        
        print(", ".join(features), ", ", action, file=open("../data/features.txt", "a+"), sep="")

        # Evaluate the action.
        if action is Action.LEFT:
            # Remove the second element on the stack.
            dependent_word = stack.pop(-2)
            # Produce a head-dependent relation
            # between between the word at the top of stack 
            # and the word below it.
            print("(", dependent_word.Form, "<--", stack[-1].Form, ")", end=endl_sep,\
              file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
            input_line.loc[input_line["ID"] == dependent_word.ID, "Head"] = stack[-1].ID
            
        elif action is Action.RIGHT:
            # Remove the top element on the stack.
            dependent_word = stack.pop(-1)
            # Produce a head-dependent relation
            # between the second word on the stack and the word at the top.
            print("(", stack[-1].Form, "-->", dependent_word.Form, ")", end=endl_sep,\
              file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
            input_line.loc[input_line["ID"] == dependent_word.ID, "Head"] = stack[-1].ID
            
        elif action is Action.SHIFT:
            # Remove the word from the front of the input buffer 
            # (i.e. increase by 1 the input buffer index)
            # and push it onto the stack.
            stack += [input_line.iloc[buffer_index, :]]
            buffer_index += 1
            print("", end=endl_sep,\
              file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
        else:
            raise ValueError("UNKNOWN ACTION:", action)
        
        step_count += 1
    
    # Print the final state.
    print(step_count, end=sep,\
      file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
    print([word.Form for word in stack], end=sep,\
      file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
    print(list(input_line.iloc[buffer_index:, 1]), end=sep,\
      file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
    print("DONE", end=sep,\
      file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
    print("", end=endl_sep + "\n",\
      file=(open(print_to_file, "a+") if print_to_file is not None else sys.stdout))
            
    
 #%% MAIN   

if __name__ == "__main__":
    
    #%% LOAD DATAs
    
    # Load the input sentences into a DataFrame
    filename = "../data/input.txt"
    lines = pd.read_csv(filename)
    
    #%% PREPROCESS DATASET
    
    # Set the dependency relation to a generic "dep"
    lines["Deprel"] = "dep"
    

    
    # Look at the universal parts of speech
    pos = set(lines.UPOS)
    print("UPOS:", pos)
    # Count their occurrencies
    pos_occ_dict = lines.UPOS.value_counts()
    print(pos_occ_dict)
    
    #%% BUILD SUB-DATAFRAMES
    
    # We want to process each sentence individually.
    # To do so, split the dataframe into sub-dataframes, and store them into an array.
    
    # Obtain the indices of where each sentence starts.
    start_indices = lines[lines.ID == 1].index.tolist()
    # Build an array where the sub-dataframes are stored.
    # Handle manually the last sentence.
    lines_array = [None] * len(start_indices)
    for i in range(len(start_indices) - 1):
        lines_array[i] = lines.iloc[start_indices[i]:start_indices[i+1], :]
    lines_array[-1] = lines.iloc[start_indices[-1]:, :]
    
    #%% PARSE
    for l in lines_array:
        dep_parser(l, oracle=rule_based_oracle, print_details=True, print_to_file="../data/conftable.txt")
    
    #%% WRTIE OUTPUT
    
    with open('../data/output.txt', 'w') as f:
        print("ID,Form,Lemma,UPOS,Head,Deprel\n", file=f)
        for l in lines_array:
            # Set the head relation to int
            l["Head"] = l["Head"].astype(int)
            l.to_csv(f, index=False, header=False)
            print("", file=f)



    
    
    
    
    

        