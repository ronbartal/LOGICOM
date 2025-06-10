import json
from type import ModelType, AgentType
import tkinter as tk
import pandas as pd
import os
import html
import shutil

def replace_variables(text, variables):
    for variable, value in variables.items():
        text = text.replace(variable, str(value))
    return text


def extract_prompt(file_path, variables):
    with open(file_path, 'r') as file:
        content = file.read()

    system_start_marker = "==== SYSTEM ===="
    user_start_marker = "==== ASSISTANT ===="

    system_start = content.find(system_start_marker) + len(system_start_marker)
    system_end = content.find(user_start_marker)
    system_text = content[system_start:system_end].strip()

    user_start = content.find(user_start_marker) + len(user_start_marker)
    assistant_text = content[user_start:].strip()

    system_text = replace_variables(system_text, variables)
    assistant_text = replace_variables(assistant_text, variables)

    return system_text, assistant_text


def save_xlsx(memory_log_path, topic_id, chat_id, helper_type, result, number_of_rounds, claim):
    memory_log_path_xlsx = os.path.join(memory_log_path, "all.xlsx")

    # Define all expected columns
    all_columns = ["Topic_ID", "claim", "No_Helper", "Vanilla_Helper", "Fallacy_Helper", 
                   "No_Helper_Round", "Vanilla_Helper_Round", "Fallacy_Helper_Round", 
                   "Chat_ID_No_Helper", "Chat_ID_Vanilla_Helper", "Chat_ID_Fallacy_Helper",
                   "Run_Number", "Moderator_Used"]

    if os.path.exists(memory_log_path_xlsx):
        df = pd.read_excel(memory_log_path_xlsx)
        
        # Ensure all columns exist
        for col in all_columns:
            if col not in df.columns:
                df[col] = None
        
        # Create unique identifier for this specific run
        run_id = f"{topic_id}_{helper_type}_{chat_id}"
        
        # Check if this exact run already exists
        existing_run = df[(df["Topic_ID"] == topic_id) & 
                         (df[f"Chat_ID_{helper_type}"] == str(chat_id))]
        
        if not existing_run.empty:
            # Update existing run
            idx = existing_run.index[0]
            df.loc[idx, helper_type] = result
            df.loc[idx, f'{helper_type}_Round'] = number_of_rounds
            df.loc[idx, f'Chat_ID_{helper_type}'] = str(chat_id)
        else:
            # Add new row for this specific run
            new_row_data = {col: None for col in all_columns}
            new_row_data.update({
                "Topic_ID": topic_id,
                "claim": claim,
                helper_type: result,
                f'{helper_type}_Round': number_of_rounds,
                f'Chat_ID_{helper_type}': str(chat_id)
            })
            new_row_df = pd.DataFrame([new_row_data])
            df = pd.concat([df, new_row_df], ignore_index=True)
    else:
        # Create new DataFrame with all columns
        new_row_data = {col: None for col in all_columns}
        new_row_data.update({
            "Topic_ID": topic_id,
            "claim": claim,
            helper_type: result,
            f'{helper_type}_Round': number_of_rounds,
            f'Chat_ID_{helper_type}': str(chat_id)
        })
        df = pd.DataFrame([new_row_data])

    # Save the DataFrame to the Excel file using XlsxWriter as the engine
    writer = pd.ExcelWriter(memory_log_path_xlsx, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()


def generate_summary_statistics(memory_log_path):
    """Generate comprehensive summary statistics from all debate results"""
    memory_log_path_xlsx = os.path.join(memory_log_path, "all.xlsx")
    
    if not os.path.exists(memory_log_path_xlsx):
        print("No results file found to analyze.")
        return
    
    df = pd.read_excel(memory_log_path_xlsx)
    
    if df.empty:
        print("Results file is empty.")
        return
    
    # Ensure Is_Valid_Run column exists
    if 'Is_Valid_Run' not in df.columns:
        # Recalculate validity for existing data
        df['Is_Valid_Run'] = df.apply(
            lambda row: is_valid_debate_run(row.get('Finish_Reason', 'unknown'), 
                                          row.get('Number_of_Rounds', 0)), axis=1
        ).astype(int)
    
    # Overall statistics (including all runs)
    total_debates = len(df)
    successful_debates = len(df[df['Result'] == True])
    overall_success_rate = (successful_debates / total_debates * 100) if total_debates > 0 else 0
    
    # Valid runs only statistics
    valid_df = df[df['Is_Valid_Run'] == 1]
    total_valid_debates = len(valid_df)
    successful_valid_debates = len(valid_df[valid_df['Result'] == True])
    valid_success_rate = (successful_valid_debates / total_valid_debates * 100) if total_valid_debates > 0 else 0
    
    # Invalid runs analysis
    invalid_df = df[df['Is_Valid_Run'] == 0]
    total_invalid_debates = len(invalid_df)
    invalid_percentage = (total_invalid_debates / total_debates * 100) if total_debates > 0 else 0
    
    # Claim-level statistics (how many unique claims were processed)
    unique_claims = df['Topic_ID'].nunique()
    runs_per_claim_helper = df.groupby(['Topic_ID', 'Helper_Type']).size().mean()
    
    # Statistics by helper type (both filtered and unfiltered)
    helper_stats = {}
    helper_types = df['Helper_Type'].unique()
    
    for helper in helper_types:
        helper_df = df[df['Helper_Type'] == helper]
        helper_valid_df = valid_df[valid_df['Helper_Type'] == helper]
        
        # All runs (unfiltered)
        total = len(helper_df)
        successful = len(helper_df[helper_df['Result'] == True])
        success_rate = (successful / total * 100) if total > 0 else 0
        
        # Valid runs only (filtered)
        total_valid = len(helper_valid_df)
        successful_valid = len(helper_valid_df[helper_valid_df['Result'] == True])
        valid_rate = (successful_valid / total_valid * 100) if total_valid > 0 else 0
        
        # Invalid runs for this helper
        helper_invalid = len(helper_df[helper_df['Is_Valid_Run'] == 0])
        invalid_rate = (helper_invalid / total * 100) if total > 0 else 0
        
        # Moderator usage statistics
        palm_used = len(helper_df[helper_df['Moderator_Used'] == 'PALM'])
        gpt4_used = len(helper_df[helper_df['Moderator_Used'] == 'GPT-4'])
        palm_failed = len(helper_df[helper_df['Moderator_Used'] == 'GPT-4 (PALM_Failed)'])
        
        helper_stats[helper] = {
            'Total_Debates': total,
            'Valid_Debates': total_valid,
            'Invalid_Debates': helper_invalid,
            'Invalid_Percentage': round(invalid_rate, 2),
            'Success_Rate_All_Runs': round(success_rate, 2),
            'Success_Rate_Valid_Only': round(valid_rate, 2), 
            'Successful_All_Runs': successful,
            'Failed_All_Runs': total - successful,
            'Successful_Valid_Runs': successful_valid,
            'Failed_Valid_Runs': total_valid - successful_valid,
            'Unique_Claims_Processed': len(helper_df['Topic_ID'].unique()),
            'Average_Runs_Per_Claim_Helper': round(helper_df.groupby('Topic_ID').size().mean(), 2)
        }
    
    # Create the main summary table - exactly matching the expected format
    main_summary = {
        'Total_Debates': total_debates,
        'Valid_Debates': total_valid_debates, 
        'Invalid_Debates': total_invalid_debates,
        'Invalid_Percentage': round(invalid_percentage, 2),
        'Success_Rate_All_Runs': round(overall_success_rate, 2),
        'Success_Rate_Valid_Only': round(valid_success_rate, 2),
        'Successful_All_Runs': successful_debates,
        'Failed_All_Runs': total_debates - successful_debates,
        'Successful_Valid_Runs': successful_valid_debates,
        'Failed_Valid_Runs': total_valid_debates - successful_valid_debates,
        'Unique_Claims_Processed': unique_claims,
        'Average_Runs_Per_Claim_Helper': round(runs_per_claim_helper, 2)
    }
    
    # Convert to DataFrame with exact column order
    main_summary_df = pd.DataFrame([main_summary])
    
    # Helper statistics DataFrame
    helper_summary_df = pd.DataFrame.from_dict(helper_stats, orient='index')
    
    # Save comprehensive statistics
    summary_path = os.path.join(memory_log_path, "comprehensive_statistics.xlsx")
    with pd.ExcelWriter(summary_path, engine='xlsxwriter') as writer:
        # Main summary table - exactly as discussed
        main_summary_df.to_excel(writer, index=False, sheet_name='Main_Summary')
        
        # Helper-by-helper breakdown
        helper_summary_df.to_excel(writer, sheet_name='Helper_Statistics')
        
        # Detailed results 
        df.to_excel(writer, index=False, sheet_name='All_Debate_Results')
        
        # Valid runs only
        if not valid_df.empty:
            valid_df.to_excel(writer, index=False, sheet_name='Valid_Runs_Only')
        
        # Invalid runs analysis
        if not invalid_df.empty:
            invalid_df.to_excel(writer, index=False, sheet_name='Invalid_Runs_Analysis')
    
    # Console output
    print(f"\nCOMPREHENSIVE STATISTICS GENERATED:")
    print(f"File saved to: {summary_path}")
    print(f"\n{'='*60}")
    print("MAIN EXPERIMENT SUMMARY") 
    print(f"{'='*60}")
    print(f"Total debates: {total_debates}")
    print(f"Valid debates: {total_valid_debates}")
    print(f"Invalid debates: {total_invalid_debates} ({invalid_percentage:.2f}%)")
    print(f"Success rate (ALL runs): {overall_success_rate:.2f}% ({successful_debates}/{total_debates})")
    print(f"Success rate (VALID runs only): {valid_success_rate:.2f}% ({successful_valid_debates}/{total_valid_debates})")
    print(f"Unique claims processed: {unique_claims}")
    print(f"Average runs per claim/helper: {runs_per_claim_helper:.2f}")
    
    print(f"\n{'='*60}")
    print("SUCCESS RATES BY HELPER TYPE")
    print(f"{'='*60}")
    print(f"{'Helper Type':<15} {'All Runs':<15} {'Valid Only':<15} {'Invalid %':<10}")
    print("-" * 65)
    for helper, stats in helper_stats.items():
        print(f"{helper:<15} {stats['Success_Rate_All_Runs']:>6.2f}% ({stats['Successful_All_Runs']:>2}/{stats['Total_Debates']:<2}) "
              f"{stats['Success_Rate_Valid_Only']:>6.2f}% ({stats['Successful_Valid_Runs']:>2}/{stats['Valid_Debates']:<2}) "
              f"{stats['Invalid_Percentage']:>6.2f}%")
    
    return {
        'main_summary': main_summary,
        'helper_stats': helper_stats
    }


def save_jason(memory_log,memory_log_path, topic_id, chat_id, helper_type, result,
             number_of_rounds,finish_reason):

    memory_log_path_json = os.path.join(memory_log_path, topic_id, helper_type, f"{chat_id}.json")
    data = {"Topic_ID": topic_id,
            "log": memory_log.chat_memory.log[0].inputs,
            "chat_id": chat_id,
            "number_of_rounds":number_of_rounds,
            "Stop_reason": finish_reason,
            "Convinced?": result,
           }
    with open(memory_log_path_json, "w") as json_file:
        json.dump(data, json_file)





def save_html(memory_log, memory_log_path, topic_id, chat_id, helper_type, result,
              number_of_rounds,finish_reason):
    """
    Save log files in HTML format.
    """
    # Define color codes
    colors = ['blue', 'green']  # Blue and green colors
    round_number_color = 'red'

    # Generate the HTML content
    html_content = '<html><head><style>'
    html_content += 'body { font-family: Arial, sans-serif; }'
    html_content += '.log-container { width: 80%; margin: 0 auto; word-wrap: break-word; white-space: pre-wrap; }'
    html_content += f'.round-number {{ color: {round_number_color}; font-weight: bold; }}'
    html_content += f'.log-entry {{ font-weight: bold; }}'
    html_content += '</style></head><body>'

    html_content += '<div class="log-container">'

    for i, item in enumerate(memory_log.chat_memory.log[0].inputs):
        color = colors[i % len(colors)]
        additional_string = 'Convincing_AI: ' if i % 2 == 0 else 'Debater_Agent:'

        # Calculate the rounded number
        round_number = (i // 2) + 1

        log_text = list(item.values())[0]

        # Escape special characters for HTML display
        log_text = html.escape(log_text)

        html_content += f'<div class="round-number">{round_number}.</div>'
        html_content += f'<div class="log-entry" style="color:{color};">{additional_string}{log_text}</div>'
        html_content += '<br>'

    html_content += '</div>'  # Close the log-container div

    # Add result and number of rounds at the end of the file
    html_content += '<br>'
    html_content += f'<div class="log-entry"><b>Result:</b> {result}</div>'
    html_content += f'<div class="log-entry"><b>Stop Reason:</b> {finish_reason}</div>'
    html_content += f'<div class="log-entry"><b>Number of Rounds:</b> {number_of_rounds}</div>'

    html_content += '</body></html>'

    memory_log_path_html = os.path.join(memory_log_path, topic_id, helper_type, f"{chat_id}.html")
    with open(memory_log_path_html, 'w') as file:
        file.write(html_content)

def save_txt(memory_log,memory_log_path, topic_id, chat_id, helper_type, result,
             number_of_rounds,finish_reason):
    """
    Save log files in txt format. Agent responses.
    """
    json_data = json.dumps(memory_log.chat_memory.log[0].inputs)
    memory_log_path_txt=os.path.join(memory_log_path, topic_id, helper_type, f"{chat_id}.txt")
    with open(memory_log_path_txt, 'w') as file:
        file.write(json_data)


def create_directory(directory_path):
    if os.path.exists(directory_path):
        # Directory exists, so remove it
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' removed successfully.")
        except OSError as e:
            print(f"Error occurred while removing directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' does not exist.")

    # Create the new directory
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except OSError as e:
        print(f"Error occurred while creating directory '{directory_path}': {e}")
def save_log(memory_log, memory_log_path, topic_id, chat_id, result, helper, number_of_rounds, claim, finish_reason, run_number=1, moderator_used="PALM"):
    """
    Save log files in both HTML and txt formats.
    """
    create_directory(os.path.join(memory_log_path, topic_id, helper))
    save_jason(memory_log=memory_log,memory_log_path=memory_log_path, topic_id=topic_id, chat_id=chat_id, helper_type=helper, result=result,
             number_of_rounds=number_of_rounds,finish_reason=finish_reason)
    save_html(memory_log=memory_log,memory_log_path=memory_log_path, topic_id=topic_id, chat_id=chat_id, helper_type=helper, result=result,
             number_of_rounds=number_of_rounds,finish_reason=finish_reason)
    save_txt(memory_log=memory_log,memory_log_path=memory_log_path, topic_id=topic_id, chat_id=chat_id, helper_type=helper, result=result,
             number_of_rounds=number_of_rounds,finish_reason=finish_reason)
    save_xlsx_enhanced(memory_log_path=memory_log_path, topic_id=topic_id, chat_id=chat_id, helper_type=helper, result=result,
             number_of_rounds=number_of_rounds, claim=claim, run_number=run_number, moderator_used=moderator_used, 
             finish_reason=finish_reason)


def save_xlsx_enhanced(memory_log_path, topic_id, chat_id, helper_type, result, number_of_rounds, claim, run_number, moderator_used, finish_reason="unknown"):
    """Enhanced Excel save function with run number, moderator tracking, and finish reason"""
    memory_log_path_xlsx = os.path.join(memory_log_path, "all.xlsx")

    # Define all expected columns
    all_columns = ["Topic_ID", "claim", "Helper_Type", "Result", "Number_of_Rounds", 
                   "Chat_ID", "Run_Number", "Moderator_Used", "Success", "Finish_Reason", "Is_Valid_Run"]

    if os.path.exists(memory_log_path_xlsx):
        df = pd.read_excel(memory_log_path_xlsx)
        
        # Ensure all columns exist
        for col in all_columns:
            if col not in df.columns:
                df[col] = None
    else:
        # Create new DataFrame with all columns
        df = pd.DataFrame(columns=all_columns)

    # Determine if this is a valid run (not off-topic, greeting loop, etc.)
    is_valid_run = is_valid_debate_run(finish_reason, number_of_rounds)

    # Add new row for this specific run
    new_row_data = {
        "Topic_ID": topic_id,
        "claim": claim,
        "Helper_Type": helper_type,
        "Result": result,
        "Number_of_Rounds": number_of_rounds,
        "Chat_ID": str(chat_id),
        "Run_Number": run_number,
        "Moderator_Used": moderator_used,
        "Success": 1 if result else 0,
        "Finish_Reason": finish_reason,
        "Is_Valid_Run": 1 if is_valid_run else 0
    }
    
    new_row_df = pd.DataFrame([new_row_data])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Save the DataFrame to the Excel file using XlsxWriter as the engine
    writer = pd.ExcelWriter(memory_log_path_xlsx, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()


def is_valid_debate_run(finish_reason, number_of_rounds):
    """
    Determine if a debate run is valid (should be included in success rate calculations)
    
    Invalid runs include:
    - Off-topic conversations (<OFF-TOPIC>)
    - Greeting loops (<TERMINATE> for greeting)  
    - Very short runs (< 3 rounds) that indicate failure to engage
    
    Valid runs include:
    - Successful persuasion (regardless of round count)
    - Safety stops at max rounds
    - Normal debate conclusions
    """
    if not finish_reason:
        return True  # Default to valid if no finish reason
    
    finish_reason_str = str(finish_reason).lower()
    
    # Always valid: Successful persuasion regardless of round count
    success_indicators = [
        'successfully convinced',
        'persuader successfully',
        'convinced the debater',
        'debater convinced',
        'persuasion successful'
    ]
    
    for success_pattern in success_indicators:
        if success_pattern in finish_reason_str:
            return True  # Always valid if persuasion was successful
    
    # Invalid run conditions
    invalid_patterns = [
        '<off-topic>',
        'off-topic',
        'greeting',
        'hello',
        'greet'
    ]
    
    # Check for invalid patterns
    for pattern in invalid_patterns:
        if pattern in finish_reason_str:
            return False
    
    # Very short runs (< 3 rounds) are usually invalid UNLESS they were successful
    # (This check comes after success check, so short successful runs are already marked valid)
    if number_of_rounds < 3:
        return False
    
    # Safety stops and other normal conclusions are considered valid
    return True


r''' Save the generated fallacy. The file address should be changed manually.  '''


def save_fallacy(topic_id, chat_id, argument, counter_argument, fallacy, fallacious_argument=None):

    path = 'path/to/save/fallacies/'
    df = pd.read_csv(path)
    if not fallacious_argument:
        fallacious_argument = ''

    new_row = pd.DataFrame([[topic_id, chat_id, argument, counter_argument, fallacy,fallacious_argument]],
                           columns=["Topic_ID", "Chat_ID", "Argument", "Counter_Argument", "Fallacy", 'Fallacious_Argument'])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)


def get_variables(data, agent_type, dataset_type=None):

    # Handling for different agent types
    if agent_type == AgentType.DEBATER_AGENT:
        return {
            "<TOPIC>": str(data['title']),
            "<CLAIM>": str(data['claim']),
            "<ORIGINAL_TEXT>": str(data['original_text']),
            "<REASON>": str(data['reason']),
            "<WARRANT_ONE>": str(data['warrant_one']),
            "<WARRANT_TWO>": str(data['warrant_two']),
            "<SIDE>": "ONE",
            "<O-SIDE>": "TWO"
        }
    elif agent_type == AgentType.PERSUADER_AGENT:
        return {
            "<TOPIC>": str(data['title']),
            "<CLAIM>": str(data['claim']),
            "<ORIGINAL_TEXT>": str(data['original_text']),
            "<REASON>": str(data['reason']),
            "<WARRANT_ONE>": str(data['warrant_one']),
            "<WARRANT_TWO>": str(data['warrant_two']),
            "<SIDE>": "TWO",
            "<O-SIDE>": "ONE"
        }


r''' This is for human persuader-test'''


class ChatWindow:
    def __init__(self, on_submit):
        self.window = tk.Tk()
        self.input_text = tk.StringVar()
        self.chat_text = tk.StringVar()
        self.on_submit = on_submit
        self.is_running = True

        self.window.title("Chatbot")
        self.window.geometry("1800x1600")

        chat_frame = tk.Frame(self.window, relief=tk.SUNKEN, bd=2)
        chat_frame.pack(pady=100, expand=True, fill=tk.BOTH)

        chat_scrollbar = tk.Scrollbar(chat_frame)
        chat_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        chat_view = tk.Text(chat_frame, yscrollcommand=chat_scrollbar.set)
        chat_view.pack(expand=True, fill=tk.BOTH)
        chat_scrollbar.config(command=chat_view.yview)

        input_frame = tk.Frame(self.window)
        input_frame.pack(pady=10)

        input_entry = tk.Entry(input_frame, textvariable=self.input_text)
        input_entry.pack(side=tk.LEFT)

        submit_button = tk.Button(input_frame, text="Submit", command=self.submit)
        submit_button.pack(side=tk.LEFT)

        break_button = tk.Button(input_frame, text="Break", command=self.break_loop)
        break_button.pack(side=tk.LEFT)

        self.chat_view = chat_view
        chat_view.tag_configure("bot", foreground="red")

        self.input_entry = input_entry  # Save the input entry reference
        self.window.mainloop()

    def submit(self):
        user_input = self.input_text.get()
        self.on_submit(user_input, self)
        self.input_text.set("")

    def break_loop(self):
        self.is_running = False
        self.window.destroy()

    def resize_input_entry(self, event):
        text = self.input_entry.get()
        width = len(text) + 1  # Calculate the new width based on the input text
        self.input_entry.config(width=width)

