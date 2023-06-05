import yaml
import streamlit as st
import streamlit_authenticator as stauth
from Auth import Auth

class Sidebar():
	def __init__(self, auth: Auth):
		self.name = auth.get_name()
		self.authentication_status = auth.get_authentication_status()
		self.username = auth.get_username()
		self.authenticator = auth.get_authenticator()
		self.max_iterations = 5

	# FUNCTION: TRY EXAMPLE
	# def try_example(self) -> str:
	# 	rand = random.randint(0, 2)
	# 	example_prompt = ""
	# 	if rand == 0:
	# 		example_prompt = "Search for the iPhone 14 Pro case with the most value."
	# 	elif rand == 1:
	# 		example_prompt = "Formulate a shopping cart with all designer items with a total less than $1000"
	# 	else:
	# 		example_prompt = "Find me a premium webcam good for video calls."
	# 	return example_prompt

	def get_max_iterations(self) -> int:
		return self.max_iterations
	
	def set_max_iterations(self, max_iterations: int):
		self.max_iterations = max_iterations

	def main(self):
		# SIDEBAR: AUTH CHECK
		if self.authentication_status is False:
			st.sidebar.error('Username/password is incorrect')
		elif self.authentication_status is None:
			st.sidebar.warning('Please enter your username and password')
			st.warning('👈 Please Use Sidebar To Login/Register To Use Calvin')


		# SIDEBAR: REGISTER & FORGOT FORMS
		st.sidebar.text(" ")
		st.sidebar.text(" ")
		if self.authentication_status is False or self.authentication_status is None:
			# REGISTER USER
			with st.sidebar.expander("📝 Register", expanded=False):
				try:
					if self.authenticator.register_user('Register user', preauthorization=False):
						st.success('User registered successfully')
						with open('config.yaml', 'w') as file:
							yaml.dump(self.config, file, default_flow_style=False)
				except Exception as e:
					st.error(e)
			with st.sidebar.expander("🤷‍♂️ Forgot Password/Username", expanded=False):
				# FORGOT PASSWORD
				try:
					username_forgot_pw, email_forgot_password, random_password = self.authenticator.forgot_password('Forgot password')
					if username_forgot_pw:
						st.success('New Temporary Password: ' + random_password)
						with open('config.yaml', 'w') as file:
							yaml.dump(self.config, file, default_flow_style=False)
					else:
						st.error('Username not found')
				except Exception as e:
					st.error(e)
				# FORGOT USERNAME
				try:
					username_forgot_username, email_forgot_username = self.authenticator.forgot_username('Forgot username')
					if username_forgot_username:
						st.success('Username: ' + username_forgot_username)
						with open('config.yaml', 'w') as file:
							yaml.dump(self.config, file, default_flow_style=False)
					else:
						st.error('Email not found')
				except Exception as e:
					st.error(e)

		# SIDEBAR: APP INFO
		if self.authentication_status:
			st.sidebar.header("Welcome " + self.name + "! 👋")
			st.sidebar.markdown(""":black[Use Calvin to help you shop!]""")
			# st.sidebar.button("⌛️ Try Example", on_click=self.try_example, type="secondary")
			with st.sidebar.expander("✍️ Prompt Examples", expanded=False):
				st.markdown(
				""" 
					:black[For Best Results, Use An Objective Format:\n]
					------
					:black[1. *"Find me a premium webcam good for video calls."*]
					:black[2. *"Search for the iPhone 14 Pro case with the most value."*]
					:black[3. *"Formulate a shopping cart with all designer items with a total less than $1000"*]
					"""
				)
			st.sidebar.progress(100)
		

		# SIDEBAR: CHAT HISTORY
		if self.authentication_status:
			st.sidebar.header("Settings")
			with st.sidebar.expander("🧠 Model Settings ", expanded=False):
				iterations = st.slider("Max Iterations", min_value=1, max_value=10, step=1, value=5)
				self.set_max_iterations(iterations)
			with st.sidebar.expander("🛠️ Account Settings", expanded=False):
				try:
					if self.authenticator.update_user_details(self.username, 'Update user details'):
						st.success('Entries updated successfully')
						with open('config.yaml', 'w') as file:
							yaml.dump(self.config, file, default_flow_style=False)
				except Exception as e:
					st.error(e)
				try:
					if self.authenticator.reset_password(self.username, 'Reset password'):
						st.success('Password modified successfully')
						with open('config.yaml', 'w') as file:
							yaml.dump(self.config, file, default_flow_style=False)
				except Exception as e:
					st.error(e)
			st.sidebar.text(" ")
			st.sidebar.progress(100)
			
		# SIDEBAR: ACCOUNT SETTINGS & LOGOUT
		if self.authentication_status:
			st.sidebar.text(" ")
			self.authenticator.logout('✌️ Logout', 'sidebar')
			