from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool ,ScrapeWebsiteTool

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class FindOwners():
	"""FindOwners crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def lead_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['lead_researcher'],
			tools=[SerperDevTool(),ScrapeWebsiteTool(website_url="https://www.synergymarinegroup.com/services/offshore-ship-management-services/")],
			verbose=True
		)

	@agent
	def review_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['review_agent'],
			tools=[SerperDevTool()],
			verbose=True
		)
	
	@agent
	def strategy_developer(self) -> Agent:
		return Agent(
			config=self.agents_config['strategy_developer'],
			tools=[SerperDevTool(),ScrapeWebsiteTool(website_url="https://www.synergymarinegroup.com/edt-synergy-marine-group-form-joint-venture/"),
		  			ScrapeWebsiteTool(website_url="https://www.edtoffshore.com/solutions-services/ship-management-operations")],
			verbose=True
		)
	
	@agent
	def process_manager(self) -> Agent:
		return Agent(
			config=self.agents_config['process_manager'],
			tools=[SerperDevTool()],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def initial_research(self) -> Task:
		return Task(
			config=self.tasks_config['initial_research'],
		)

	@task
	def lead_review(self) -> Task:
		return Task(
			config=self.tasks_config['lead_review']
		)
	
	@task
	def strategy_formulation(self) -> Task:
		return Task(
			config=self.tasks_config['strategy_formulation'],
		)
	
	@task
	def process_oversight(self) -> Task:
		return Task(
			config=self.tasks_config['process_oversight'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the FindOwners crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.hierarchical,
			manager_llm= LLM(model='azure/gpt-4o-2024-11-20'),
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
