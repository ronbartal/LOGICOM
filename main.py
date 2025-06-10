import os

# Set OpenAI API key from the API_keys file
try:
    with open('API_keys', 'r') as f:
        for line in f:
            if 'OpenAI_API_key:' in line:
                os.environ["OPENAI_API_KEY"] = line.split('OpenAI_API_key:')[1].strip()
            elif 'Google_API_key:' in line:
                os.environ["GOOGLE_API_KEY"] = line.split('Google_API_key:')[1].strip()
except Exception as e:
    print(f"Error loading API keys: {e}")

import time
import openai

from tqdm import tqdm
import google.generativeai as genai
from agents.modertorAgent import ModeratorAgent
from config.gptconfig import ChatGPTConfig

from utils import *
import argparse
from agents.persuaderAgent import PersuaderAgent
from agents.debaterAgent import DebaterAgent
from colorama import init, Fore, Back, Style

import uuid

init()


def define_arguments():
    parser = argparse.ArgumentParser(description="Your script description here")

    parser.add_argument("--api_key_openai", required=False, help="API key for openAI (optional, will read from API_keys file)")
    parser.add_argument("--api_key_palm", required=False, help="API key for PaLM (optional, will read from API_keys file)")
    parser.add_argument("--claim_number", type=int, default=499, help="The last claim number in dataset (0-based indexing)")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs per helper type per claim")
    parser.add_argument("--force_gpt4_moderator", action='store_true', help="Force use of GPT-4 moderator instead of starting with PALM")
    parser.add_argument("--run_all_helpers", action='store_true', help="Run all three helper types automatically")
    parser.add_argument("--persuader_instruction", default='persuader_claim_reason_instruction',
                        help="Instruction for persuader")
    parser.add_argument("--debater_instruction", default='debater_claim_reason_instruction',
                        help="Instruction for debater")
    parser.add_argument("--helper_prompt_instruction", default='No_Helper', help="Instruction for Helper")
    parser.add_argument("--data_path", default='./claims/all-claim-not-claim.csv', help="Path to the source data")
    parser.add_argument("--log_html_path", default='./debates/', help="Path to save the debates")

    args = parser.parse_args()

    args.moderator_terminator_instruction_palm = 'moderator_terminator_instruction'
    args.moderator_tag_checker_instruction_palm = 'moderator_TagChecker_instruction'
    args.moderator_topic_checker_instruction_palm = 'moderator_topic_instruction'
    args.moderator_terminator_instruction_gpt = 'moderator_terminator_instruction'
    args.moderator_tag_checker_instruction_gpt = 'moderator_TagChecker_instruction'
    args.moderator_topic_checker_instruction_gpt = 'moderator_topic_instruction'
    args.memory_instruction = 'summary_instruction'

    return args


def run(persuader_agent, debater_agent, moderator_agent_1, moderator_agent_2, moderator_agent_3, moderator_agent_4,
        moderator_agent_5, args, topic_id, chat_id, claim, gpt4_moderator, run_number=1):
    print(arg.log_html_path)
    human = False
    keep_talking = True
    round_of_conversation: int = 1
    moderator_used = "GPT-4" if gpt4_moderator else "PALM"
    palm_failed = False
    finish_reason = "unknown"
    
    if gpt4_moderator:
        print(Fore.RED + '\n ****** GPT-4_MODERATOR *******' + Style.RESET_ALL)
    else:
        print(Fore.GREEN + '\n ****** PaLM_MODERATOR *******' + Style.RESET_ALL)
    print(Fore.RED + '\n ******************************* TOPIC******************' + Style.RESET_ALL)
    print(Fore.RED + '******************************* TOPIC******************' + Style.RESET_ALL)
    print(Fore.RED + '******************************* TOPIC******************' + Style.RESET_ALL)
    print(f"Run {run_number}/3 - Topic ID: {topic_id}")
    print(persuader_agent.last_response)

    if human:
        def handle_submit(user_input, window):
            debater_response_to_human = debater_agent.call(user_input)
            window.chat_view.insert(tk.END, "You: {}\n".format(user_input))
            window.chat_view.insert(tk.END, "\n")
            window.chat_view.insert(tk.END, "-" * 30 + "\n", "line")
            window.chat_view.insert(tk.END, "Debater: {}\n".format(debater_response_to_human))
            window.chat_view.insert(tk.END, "\n")
            window.chat_view.see(tk.END)

            if not window.is_running:
                window.window.destroy()

        chat_window = ChatWindow(on_submit=handle_submit)
        chat_window.insert(tk.END, "You: {}\n".format('test'))
        chat_window.see(tk.END)
    else:

        while keep_talking:
            result = None

            r"""debater"""
            print(Fore.MAGENTA + 'round_of_conversation:' + str(round_of_conversation) + Style.RESET_ALL)
            print(Fore.GREEN + "\033[3m ********Agent2*************.\033[0m " + Style.RESET_ALL)
            print(Fore.GREEN + "********Debater*************" + Style.RESET_ALL)
            debater_response = debater_agent.call(persuader_agent.last_response)
            print(debater_response)

            r"""persuader"""
            print(Fore.BLUE + "\033[3m ********Agent1*************.\033[0m " + Style.RESET_ALL)
            print(Fore.BLUE + "********Persuader*************" + Style.RESET_ALL)
            if persuader_agent.helper_feedback_switch:
                persuader_response, fallacy, fallacious_argument = persuader_agent.call(debater_agent.last_response)
            else:
                persuader_response, fallacy = persuader_agent.call(debater_agent.last_response)
                fallacious_argument = None
            print(persuader_response)

            r'''**********Moderator*********'''
            print(Fore.RED + "******** MODERATOR *************" + Style.RESET_ALL)
            # This is used for topics that PaLM sends None.
            if gpt4_moderator:
                print(Fore.RED + '\n ****** GPT-4_MODERATOR *******' + Style.RESET_ALL)
                # GPT 4 Moderator
                moderator_command_4 = moderator_agent_4.call(persuader_agent.memory.chat_memory.log[0].inputs)
                # GPT 3 MODERATOR
                # moderator_command_5 = moderator_agent_5.call(persuader_agent.memory.chat_memory.log[0].inputs)
                print(moderator_command_4.info.value)
                if moderator_command_4.result:
                    result = True
                else:
                    result = False
                if moderator_command_4.terminate:
                    finish_reason = str(moderator_command_4.info.value)
                    break
                time.sleep(1)
            else:

                moderator_command_1 = moderator_agent_1.call(persuader_agent.memory.chat_memory.log[0].inputs)

                if moderator_command_1 == None:
                    print(
                        Fore.RED + '\n ****** Replacing PaLM moderator with GPT-4_MODERATOR *******' + Style.RESET_ALL)
                    moderator_command_4 = moderator_agent_4.call(persuader_agent.memory.chat_memory.log[0].inputs)
                    # GPT 3 MODERATOR
                    #  moderator_command_5 = moderator_agent_5.call(persuader_agent.memory.chat_memory.log[0].inputs)
                    print(moderator_command_4.info.value)
                    if moderator_command_4.result:
                        result = True
                    else:
                        result = False
                    if moderator_command_4.terminate:
                        finish_reason = str(moderator_command_4.info.value)
                        break

                    time.sleep(1)
                    gpt4_moderator = True
                    moderator_used = "GPT-4 (PALM_Failed)"
                    palm_failed = True
                    continue
                moderator_command_2 = moderator_agent_2.call(persuader_agent.memory.chat_memory.log[0].inputs)
                moderator_command_3 = moderator_agent_3.call(persuader_agent.memory.chat_memory.log[0].inputs)

                print(moderator_command_1.info.value)
                print(moderator_command_2.info.value)
                print(moderator_command_3.info.value)

                if moderator_command_1.result and moderator_command_2.result and moderator_command_3.result:
                    result = True
                else:
                    result = False

                if moderator_command_1.terminate and moderator_command_2.terminate and moderator_command_3.terminate:
                    finish_reason = str(moderator_command_1.info.value)
                    break

            round_of_conversation += 1
            if int(round_of_conversation) > 10:
                print("safety stop - reached 10 rounds maximum")
                finish_reason = "safety_stop_max_rounds"
                break

            time.sleep(1)

    print(Fore.RED + "******** Conversation Result *************" + Style.RESET_ALL)
    print(f"Result: {result}")
    print(f"Moderator Used: {moderator_used}")
    print(f"Run: {run_number}/3")
    print(f"Finish Reason: {finish_reason}")

    r"""Number of token used """
    print("persuader used token: ", int(persuader_agent.model_backbone.token_used)
          + int(persuader_agent.model_backbone_helper.token_used) +
          int(persuader_agent.memory.model_backbone.token_used))
    print("Debater used token: ", int(debater_agent.model_backbone.token_used)
          +
          int(debater_agent.memory.model_backbone.token_used)
          )
    print("Number of token used for this conversation: ",
          int(debater_agent.model_backbone.token_used)
          + int(persuader_agent.model_backbone.token_used) +
          int(persuader_agent.model_backbone_helper.token_used) +
          int(debater_agent.memory.model_backbone.token_used) +
          int(persuader_agent.memory.model_backbone.token_used)
          )
    if gpt4_moderator:
        print("Number of token each moderator used", moderator_agent_4.model_backbone.token_used)
    else:
        print("Number of token each moderator used", moderator_agent_1.model_backbone.token_used)

    r"""Save the log"""
    save_log(memory_log=persuader_agent.memory,
             memory_log_path=args.log_html_path,
             topic_id=topic_id,
             chat_id=chat_id,
             result=result,
             helper=args.helper_prompt_instruction,
             number_of_rounds=round_of_conversation,
             claim=claim,
             finish_reason=finish_reason,
             run_number=run_number,
             moderator_used=moderator_used
             )


def main(arg):
    # Load API keys if not provided via command line
    if not arg.api_key_openai or not arg.api_key_palm:
        print("API keys not provided via command line, reading from API_keys file...")
        
        # API keys are already loaded at the top of the file into environment variables
        # Just verify they were loaded successfully
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OpenAI API key not found in API_keys file or environment!")
            print("Please ensure API_keys file contains: OpenAI_API_key: your_key_here")
            return
            
        if not os.environ.get("GOOGLE_API_KEY"):
            print("ERROR: Google API key not found in API_keys file or environment!")
            print("Please ensure API_keys file contains: Google_API_key: your_key_here")
            return
            
        print("✅ API keys successfully loaded from API_keys file")
    else:
        # If provided via command line, update environment variables
        if arg.api_key_openai:
            os.environ["OPENAI_API_KEY"] = arg.api_key_openai
        if arg.api_key_palm:
            os.environ["GOOGLE_API_KEY"] = arg.api_key_palm
        print("✅ API keys loaded from command line arguments")

    data = pd.read_csv(arg.data_path)
    
    # Determine which helper types to run
    if arg.run_all_helpers:
        helper_types = ['No_Helper', 'Vanilla_Helper', 'Fallacy_Helper']
        print(f"Running all helper types: {helper_types}")
    else:
        helper_types = [arg.helper_prompt_instruction]
        print(f"Running single helper type: {helper_types[0]}")
    
    # Calculate total number of debates
    num_claims = arg.claim_number + 1  # 0-based indexing, so +1 for total count
    total_debates = num_claims * len(helper_types) * arg.num_runs
    completed_debates = 0
    
    print(f"\n=== DEBATE EXPERIMENT CONFIGURATION ===")
    print(f"Number of claims: {num_claims} (0 to {arg.claim_number})")
    print(f"Helper types: {helper_types}")
    print(f"Runs per helper type: {arg.num_runs}")
    print(f"Total debates to run: {total_debates}")
    print(f"Forced GPT-4 moderator: {arg.force_gpt4_moderator}")
    print(f"Max rounds per debate: 10")
    print("==========================================\n")
    
    # Track overall statistics
    overall_stats = {helper: {'success': 0, 'total': 0, 'palm_used': 0, 'gpt4_used': 0, 'palm_failed': 0} 
                    for helper in helper_types}

    r''' Iterate through dataset and start conversations'''
    for i in tqdm(range(num_claims), desc="Processing claims"):
        
        print(f"\n{'='*60}")
        print(f"PROCESSING CLAIM {i+1}/{num_claims} (ID: {data.loc[i]['id']})")
        print(f"{'='*60}")
        
        for helper_type in helper_types:
            print(f"\n--- Helper Type: {helper_type} ---")
            
            for run_num in range(1, arg.num_runs + 1):
                print(f"\nRun {run_num}/{arg.num_runs} for helper '{helper_type}' on claim {i}")
                
                # Update progress
                completed_debates += 1
                progress_percent = (completed_debates / total_debates) * 100
                print(f"Overall Progress: {completed_debates}/{total_debates} ({progress_percent:.1f}%)")

                r''' Initialize the Moderator Agents'''
                moderator_agent_1 = ModeratorAgent(model=ModelType.GEMINI_1_5_FLASH_8B,
                                                   prompt_instruction_path_moderator_terminator='prompts/moderator/%s.txt' % arg.moderator_terminator_instruction_palm,
                                                   prompt_instruction_path_moderator_tag_checker='prompts/moderator/%s.txt' % arg.moderator_tag_checker_instruction_palm,
                                                   prompt_instruction_path_moderator_topic_checker='prompts/moderator/%s.txt' % arg.moderator_topic_checker_instruction_palm,
                                                   variables=get_variables(data.loc[i], AgentType.DEBATER_AGENT))
                                                   
                moderator_agent_2 = ModeratorAgent(model=ModelType.GEMINI_1_5_FLASH_8B,
                                                   prompt_instruction_path_moderator_terminator='prompts/moderator/%s.txt' % arg.moderator_terminator_instruction_palm,
                                                   prompt_instruction_path_moderator_tag_checker='prompts/moderator/%s.txt' % arg.moderator_tag_checker_instruction_palm,
                                                   prompt_instruction_path_moderator_topic_checker='prompts/moderator/%s.txt' % arg.moderator_topic_checker_instruction_palm,
                                                   variables=get_variables(data.loc[i], AgentType.DEBATER_AGENT))

                moderator_agent_3 = ModeratorAgent(model=ModelType.GEMINI_1_5_FLASH_8B,
                                                   prompt_instruction_path_moderator_terminator='prompts/moderator/%s.txt' % arg.moderator_terminator_instruction_palm,
                                                   prompt_instruction_path_moderator_tag_checker='prompts/moderator/%s.txt' % arg.moderator_tag_checker_instruction_palm,
                                                   prompt_instruction_path_moderator_topic_checker='prompts/moderator/%s.txt' % arg.moderator_topic_checker_instruction_palm,
                                                   variables=get_variables(data.loc[i], AgentType.DEBATER_AGENT))
                                                   
                moderator_agent_4 = ModeratorAgent(model=ModelType.GPT_4_TURBO_0613,
                                                   prompt_instruction_path_moderator_terminator='prompts/moderator/%s.txt' % arg.moderator_terminator_instruction_gpt,
                                                   prompt_instruction_path_moderator_tag_checker='prompts/moderator/%s.txt' % arg.moderator_tag_checker_instruction_gpt,
                                                   prompt_instruction_path_moderator_topic_checker='prompts/moderator/%s.txt' % arg.moderator_topic_checker_instruction_gpt,
                                                   variables=get_variables(data.loc[i], AgentType.DEBATER_AGENT))
                                                   
                moderator_agent_5 = ModeratorAgent(model=ModelType.GPT_3_5_TURBO_0125,
                                                   prompt_instruction_path_moderator_terminator='prompts/moderator/%s.txt' % arg.moderator_terminator_instruction_gpt,
                                                   prompt_instruction_path_moderator_tag_checker='prompts/moderator/%s.txt' % arg.moderator_tag_checker_instruction_gpt,
                                                   prompt_instruction_path_moderator_topic_checker='prompts/moderator/%s.txt' % arg.moderator_topic_checker_instruction_gpt,
                                                   variables=get_variables(data.loc[i], AgentType.DEBATER_AGENT))

                r''' Initialize the Persuader Agent Model Config'''
                persuader_agent_model_config = ChatGPTConfig(temperature=1.0,
                                                             presence_penalty=0.0,
                                                             frequency_penalty=0.0)
                
                # Set helper feedback switch based on current helper type
                helper_feedback_switch = (helper_type != 'No_Helper')

                r''' Initialize the Persuader Agent'''
                persuader_agent = PersuaderAgent(
                    model=ModelType.GPT_3_5_TURBO_0125,
                    model_helper=ModelType.GPT_3_5_TURBO_0125,
                    model_config=persuader_agent_model_config,
                    helper_prompt_instruction_path='prompts/helper/%s.txt' % helper_type,
                    prompt_instruction_path='prompts/persuader/%s.txt' % arg.persuader_instruction,
                    variables=get_variables(data.loc[i], AgentType.PERSUADER_AGENT),
                    memory_prompt_instruction_path='prompts/summary/%s.txt' % arg.memory_instruction,
                    helper_feedback=helper_feedback_switch)

                # Validate helper configuration
                if helper_type != 'No_Helper':
                    if not persuader_agent.helper_feedback_switch:
                        print(f'ERROR: Helper feedback not enabled for {helper_type}')
                        continue
                else:
                    if persuader_agent.helper_feedback_switch:
                        print(f'ERROR: Helper feedback enabled when it should be disabled for {helper_type}')
                        continue

                r''' Initialize the Debater Agent Model Config'''
                debater_agent_model_config_gpt = ChatGPTConfig(temperature=1.0,
                                                               presence_penalty=0.0,
                                                               frequency_penalty=0.0)

                r''' Initialize the Debater Agent'''
                debater_agent = DebaterAgent(
                    model=ModelType.GPT_3_5_TURBO_0125,
                    model_config=debater_agent_model_config_gpt,
                    prompt_instruction_path='prompts/debater/%s.txt' % arg.debater_instruction,
                    variables=get_variables(data.loc[i], AgentType.DEBATER_AGENT),
                    memory_prompt_instruction_path='prompts/summary/%s.txt' % arg.memory_instruction)
                
                # Generate unique chat ID for this run
                chat_id = uuid.uuid1()
                
                # Temporarily update args to use current helper type
                original_helper = arg.helper_prompt_instruction
                arg.helper_prompt_instruction = helper_type
                
                # Run the debate
                try:
                    run(persuader_agent, debater_agent, moderator_agent_1, moderator_agent_2, moderator_agent_3, 
                        moderator_agent_4, moderator_agent_5, arg, str(data.loc[i]['id']), 
                        str(chat_id), str(data.loc[i]['claim']), arg.force_gpt4_moderator, run_num)
                    
                    # Update statistics (this is a placeholder - actual result tracking would need to be passed back from run function)
                    overall_stats[helper_type]['total'] += 1
                    
                except Exception as e:
                    print(f"ERROR in debate run: {e}")
                    continue
                finally:
                    # Restore original helper setting
                    arg.helper_prompt_instruction = original_helper

                time.sleep(1)  # Brief pause between runs
                
            print(f"Completed all {arg.num_runs} runs for helper '{helper_type}' on claim {i}")
    
    # Generate final summary statistics
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED - GENERATING SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    try:
        summary_stats = generate_summary_statistics(arg.log_html_path)
        print("\nDetailed statistics saved to comprehensive_statistics.xlsx")
    except Exception as e:
        print(f"Error generating summary statistics: {e}")
    
    print(f"\nTotal debates completed: {completed_debates}/{total_debates}")
    print("Experiment finished successfully!")


if __name__ == '__main__':
    arg = define_arguments()
    # Using environment variable for Google API key instead of command-line argument
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    main(arg)
