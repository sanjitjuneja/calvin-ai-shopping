import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

class Auth:
	def __init__(self):
		with open('config.yaml') as file:
			self.config = yaml.load(file, Loader=SafeLoader)

		self.authenticator = stauth.Authenticate(
			self.config['credentials'],
			self.config['cookie']['name'],
			self.config['cookie']['key'],
			self.config['cookie']['expiry_days'],
			self.config['preauthorized']
		)

		self.name, self.authentication_status, self.username = self.authenticator.login('Login', 'sidebar')
	
	def get_authenticator(self):
		return self.authenticator

	def get_name(self):
		return self.name
	
	def get_authentication_status(self):
		return self.authentication_status
	
	def get_username(self):
		return self.username
	
	def get_config(self):
		return self.config