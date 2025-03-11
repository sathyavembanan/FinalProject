from pydantic_ai import Agent
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai.models.groq import GroqModel
from amadeus import Client, ResponseError
import nest_asyncio
import imaplib
import email
from email.header import decode_header
import re
import requests
from datetime import datetime, timedelta
from typing import Optional, Tuple,List,Dict
import spacy
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import asyncio
import streamlit as st
import pandas as pd
from enum import Enum
import random
from pydantic.networks import HttpUrl 


nest_asyncio.apply()

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Amadeus API credentials
AMADEUS_API_KEY = os.environ.get("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.environ.get("AMADEUS_API_SECRET")

# Initialize Amadeus Client
amadeus = Client(client_id="6J4YZ0p04PGGLfgVlSG12tbUoACJRmhx", client_secret="TpQ3l1yZRgGGLcwz")


class FlightData(BaseModel):
    airline: str
    price: float
    currency: str
    departure_time: str
    arrival_time: str
    travel_class:str
    

class FlightSearchResult(BaseModel):
    origin: str
    destination: str
    date: str
    flights: list[FlightData]
    message: str
    cheap_flight: FlightData | None = None
    lowest_price_flight: FlightData | None = None
    flight_type : FlightData | None = None

# Define a Pydantic model
class EmailContent(BaseModel):
    subject: Optional[str]
    sender: Optional[str]
    body: Optional[str]
    origin: Optional[str]
    destination: Optional[str]
    date: Optional[str]

class HotelSearchResult(BaseModel):
    city: str
    city_code: str
    check_in: str
    check_out: str
    travelers: int
    price : float
    message: str

class Carsearchresult(BaseModel):
    destination: str
    start_date: str
    end_date: str
    people_count: int
    car_type: str
    daily_rate: float  # Price per da

# Pydantic model for experience validation
class Experience(BaseModel):
    name: str
    description: Optional[str] = "No description available."
    # price: Optional[float] = None
    # currency: Optional[str] = "USD"
    image_url: Optional[HttpUrl] = None
    booking_link: Optional[HttpUrl] = None

class ExperienceCategory(str, Enum):
    RESTAURANT = "Restaurant"
    HOTEL = "Hotel"
    ATTRACTION = "Attraction"
    OTHER = "Other"


class CategorizedExperience(BaseModel):
    category: ExperienceCategory
    experiences: List[Experience]

class CategorizationResult(BaseModel):
    result_type: str = "Experience Categorization"
    categorized_experiences: Dict[ExperienceCategory, List[Experience]]


def connect_to_email(username: str, password: str, server: str = "imap.gmail.com"):
    """Connect to the email server and log in."""
    mail = imaplib.IMAP4_SSL(server)
    mail.login(username, password)
    return mail

def fetch_latest_email(mail):
    """Fetch the latest email."""
    mail.select("inbox")
    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()
    if not email_ids:
        return None
    latest_email_id = email_ids[-1]
    status, data = mail.fetch(latest_email_id, "(RFC822)")
    if status != "OK":
        return None
    return email.message_from_bytes(data[0][1])

def extract_email_address(sender: str) -> Optional[str]:
    """Extract the email address from the sender's field."""
    match = re.search(r'<(.+)>', sender)
    if match:
        senderemail = match.group(1)
        print("Extracted Email Address:", senderemail)
        return senderemail  # Return the email address found within <>
    return sender  # Return the full sender if no email address is found in <>

def parse_email(raw_email):
    """Parse the raw email using spaCy and Pydantic."""
    # Extract subject
    subject = decode_header(raw_email["Subject"])[0][0]
    if isinstance(subject, bytes):
        subject = subject.decode()

    # Extract sender
    sender = raw_email.get("From")

    # Call extract_email_address to extract and print the sender's email
    sender_email = extract_email_address(sender)

    # Extract email body
    if raw_email.is_multipart():
        for part in raw_email.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in content_disposition:
                body = part.get_payload(decode=True).decode()
                break
    else:
        body = raw_email.get_payload(decode=True).decode()

    # Process with spaCy (example: named entity recognition)
    doc = nlp(body)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Extract GPE entities and date
    origin, destination, date = extract_travel_info(entities)

    # Validate and structure the data with Pydantic
    email_data = EmailContent(
        subject=subject,
        sender=sender_email,  # Store the extracted email address
        body=body,
        origin=origin,
        destination=destination,
        date=date
    )

    # Return the parsed email data along with the extracted entities
    return email_data, entities

def extract_travel_info(entities: list) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract the origin, destination, and date from the list of entities."""
    origin = None
    destination = None
    date = None

    gpe_count = 0
    for entity, label in entities:
        if label == "GPE":
            if gpe_count == 0:
                origin = entity
                gpe_count += 1
            elif gpe_count == 1:
                destination = entity
                gpe_count += 1
        elif label == "DATE" and not date:
            date = entity

    return origin, destination, date


if __name__ == "__main__":
    username = "airagenthelper@gmail.com"
    password =  "nain bmym cfvp syjm"

    # Connect to the email server
    mail = connect_to_email(username, password)

    # Fetch the latest email
    raw_email = fetch_latest_email(mail)
    if raw_email:
        email_data, entities = parse_email(raw_email)

        # Print out the parsed email data
        print("Subject:", email_data.subject)
        print("Sender's Email Address:", email_data.sender)
       
        print("Origin:", email_data.origin)
        print("Destination:", email_data.destination)
        print("Date:", email_data.date)



    # Print extracted entities
        print("Extracted Entities:", entities)
       
    else:
        print("No emails found.")

    # Logout
    mail.logout()
  
# Initialize Amadeus API client
amadeus = Client(
    client_id="6J4YZ0p04PGGLfgVlSG12tbUoACJRmhx",  # Replace with your Amadeus API Key
    client_secret="TpQ3l1yZRgGGLcwz"  # Replace with your Amadeus API Secret
)
def get_access_token() -> str:
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": "vmXAJ683T1OOwYBXDIsRyRAjUMSjc0AF",  # Replace with your Amadeus API Key
        "client_secret": "c8Cyr3Eu5vX9Hmbf"  # Replace with your Amadeus API Secret
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(f"Failed to fetch access token: {response.status_code}, {response.text}")
        

# Modified function to prioritize API lookup for city codes
def get_city_code_from_name(city_name):
    """
    Uses the Amadeus API to convert a city name to its corresponding city code.
    Returns the city code or None if city name is not found.
    """
    # First try the API
    try:
        access_token = get_access_token()  # Reuse your existing function
        
        # Search for the city using the Amadeus API
        api_url = "https://test.api.amadeus.com/v1/reference-data/locations"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "subType": "CITY",
            "keyword": city_name,
            "page[limit]": 1  # Just get the top result
        }
        
        response = requests.get(api_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and "data" in data and len(data["data"]) > 0:
                # Get the first result's city code
                return data["data"][0]["iataCode"]
            else:
                st.warning(f"No results found for '{city_name}' in the Amadeus API.")
        else:
            st.warning(f"API returned status code {response.status_code}. Trying alternative method.")
            
        # Only use fallback if API fails completely
        return None
        
    except Exception as e:
        st.error(f"Error looking up city code via API: {e}")
        return None
    

# Get coordinates directly from API using city name
def get_coordinates_from_city_name(city_name):
    """
    Uses Amadeus API to get latitude and longitude coordinates for a city name.
    Returns a tuple of (latitude, longitude) or None if city name is not found.
    """
    try:
        access_token = get_access_token()  # Reuse your existing function
        
        # Search for the city using the Amadeus API
        api_url = "https://test.api.amadeus.com/v1/reference-data/locations"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "subType": "CITY",
            "keyword": city_name,
            "page[limit]": 1  # Just get the top result
        }
        
        response = requests.get(api_url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and "data" in data and len(data["data"]) > 0:
                # Extract coordinates
                location_data = data["data"][0]
                latitude = location_data.get("geoCode", {}).get("latitude")
                longitude = location_data.get("geoCode", {}).get("longitude")
                
                if latitude and longitude:
                    return (str(latitude), str(longitude))
        
        return None
        
    except Exception as e:
        st.error(f"Error looking up city coordinates: {e}")
        return None
        

def fetch_destination_experiences(latitude: str, longitude: str) -> List[Experience]:
    try:
        response = amadeus.shopping.activities.get(latitude=latitude, longitude=longitude)
        experiences = []
        
        for item in response.data:
            experiences.append(Experience(
                name=item.get("name", "Unknown Experience"),
                description=item.get("description", "No description provided."),
                # price=item.get("price", {}).get("amount"),
                # currency=item.get("price", {}).get("currency"),
                image_url=item["pictures"][0] if "pictures" in item and item["pictures"] else None,
                booking_link=item.get("self", {}).get("href")
            ))
        
        return experiences
    
    except ResponseError as e:
        st.error(f"Error fetching data: {e}")
        return []


        
def get_airline_name(airline_code: str) -> str:
    try:
        access_token = get_access_token()  # Fetch the token
        api_url = f"https://test.api.amadeus.com/v1/reference-data/airlines?airlineCodes={airline_code}"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data and "data" in data and len(data["data"]) > 0:
                return data["data"][0].get("commonName", airline_code)
        print(f"Failed to fetch airline name for code {airline_code}. Status: {response.status_code}")
        return airline_code
    except json.JSONDecodeError as e:
        print(f"Error looking up airline name: {e}")
        print("Failed to decode JSON:", e)
        return airline_code

def get_city_code(city_name: str) -> str:
    try:
        access_token = get_access_token()  # Fetch the token
        api_url = f"https://test.api.amadeus.com/v1/reference-data/locations"
        params = {
            "keyword": city_name,
            "subType": "CITY"
        }
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(api_url, headers=headers,params=params)
        if response.status_code == 200:
            data = response.json()
            if data and "data" in data and len(data["data"]) > 0:
               # Fetch the first matching city's IATA code
                city_code = data["data"][0].get("iataCode")
                return city_code
        else:
            print(f"Failed to fetch city code for {city_name}. Status: {response.status_code}")
    except Exception as e:
        print(f"Error looking up city code: {e}")
    return None

# Function to Get Hotels Based on City Code
def get_hotels(city_code: str, min_price: int, max_price: int) -> list:
    access_token = get_access_token()
    if not access_token:
        return []

    api_url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
    params = {"cityCode": city_code}
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(api_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        hotels = [
            {
                "hotel_id": hotel.get("hotelId", "Unknown"),
                "name": hotel.get("name", "Unknown"),
                "address": hotel.get("address", {}).get("lines", ["N/A"])[0],
                "rating": random.randint(1, 5),  # Random rating between 1 and 5
                "price": random.randint(min_price, max_price)  # Random price between min_price and max_price
            }
            for hotel in data.get("data", [])
            # if hotel.get("price", {}).get("total", 0) >= 0 and hotel.get("price", {}).get("total", 0) >= min_price
            # and hotel.get("price", {}).get("total", 0) <= max_price
            if min_price <= random.randint(min_price, max_price) <= max_price
        ]
        return hotels
    else:
        st.error(f"Failed to fetch hotel data: {response.status_code}, {response.text}")
    
    return []

# Function to Get Hotel Offers based on Hotel ID, Check-In, and Check-Out
def get_hotel_offers(hotel_id: str, check_in: str, check_out: str, travelers: int,price:float, board_type: str) -> list:
    access_token = get_access_token()
    if not access_token:
        return []

    api_url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"

    # Debugging: Print out the hotel ID being passed
    st.write(f"Selected Hotel ID: {hotel_id}")

    if not hotel_id:
        st.error("Hotel ID is missing or invalid.")
        return []
        
    params = {
        "hotelIds": hotel_id, 
        "checkInDate": check_in, 
        "checkOutDate": check_out,
        "adults": travelers , # Adding traveler count (adults)
        "price":price,
        "boardType": board_type  # Adding board type filter
    }
    headers = {"Authorization": f"Bearer {access_token}"}

    response = requests.get(api_url, headers=headers, params=params)

    # Debug: Print the full API response
    st.write("API Response:", response.text)  

    if response.status_code == 200:
        data = response.json()

        if "data" not in data or not data["data"]:
            st.warning(f"No offers available for hotel ID: {hotel_id} on {check_in} to {check_out}.")
            return []

        offers = []
        for hotel in data["data"]:
            for offer in hotel.get("offers", []):
                offers.append({
                    "checkInDate": offer.get("checkInDate", "N/A"),
                    "checkOutDate": offer.get("checkOutDate", "N/A"),
                    "price": offer.get("price", {}).get("total", "N/A"),
                    "currency": offer.get("price", {}).get("currency", "N/A"),
                    "room_type": offer.get("room", {}).get("typeEstimated", {}).get("category", "N/A"),
                    "boardType": offer.get("boardType", "N/A"),
                    "mealPlan": offer.get("room", {}).get("description", {}).get("text", "Not specified")
                })

        return offers

    elif response.status_code == 400:
        # Handle the 'no rooms available' error more gracefully
        error_data = response.json()
        if any(error["title"] == "NO ROOMS AVAILABLE AT REQUESTED PROPERTY" for error in error_data.get("errors", [])):
            st.warning("No rooms available for the selected hotel and dates.")
        else:
            st.error(f"Failed to fetch hotel offers: {response.status_code}, {response.text}")
    else:
        st.error(f"Failed to fetch hotel offers: {response.status_code}, {response.text}")
    
    return []



# Correct format for JSON (use `null` instead of `None`)
json_string = '{"execution_count": null}'

# This will work because the JSON is valid
data = json.loads(json_string)

# Now, you can safely work with the data
print(data)

#convert EUR to USD
def convert_prices_to_usd(flights: list[dict], conversion_rate: float = 1.10) -> list[dict]:
    for flight in flights:
        if flight["currency"] == "EUR":
            flight["price"] = round(flight["price"] * conversion_rate, 2)
            flight["currency"] = "USD"
    return flights

def calculate_total_time_in_hours(departure: str, arrival: str)->float:
    departure_time = datetime.strptime(departure, "%Y-%m-%dT%H:%M:%S")
    arrival_time = datetime.strptime(arrival, "%Y-%m-%dT%H:%M:%S")
    total_time = arrival_time - departure_time
    total_hours = round(total_time.total_seconds() / 3600,2)  # Convert total time to hours
    return total_hours  # Return total time in hours rounded to 2 decimal places


# # Function to check if the date falls in the 2nd or 3rd week of the month
# def is_in_week(date_obj: datetime, week_number: int) -> bool:
#     """Check if a date falls in the given week (2nd or 3rd) of the month."""
#     first_day = date_obj.replace(day=1)
#     first_weekday = first_day.weekday()

#     start_day = (week_number - 1) * 7 + 1 - first_weekday
#     start_day = max(start_day, 1)
#     end_day = start_day + 6

#     return start_day <= date_obj.day <= end_day
    

def filter_and_sort_flights_by_price_and_duration(flights: list[dict]) -> list[dict]:
    # Filter flights for direct ones
    cheap_flight = [flight for flight in flights if flight["num_stops"] == 0]

     # If no direct flights, consider flights with at least one stop
    if not cheap_flight:
        cheap_flight = flights  # Include all flights instead of only direct ones
    
  # Sort first by price, then by total flight time
    return sorted(cheap_flight, key=lambda x: (x["price"], x["total_time_hours"]))

def direct_indirect_flight(num_stops: int) -> str:

    flight_type= lambda num_stops: "Direct" if num_stops == 0 else "Indirect"
    
    return flight_type(num_stops)

car_data = {
    ('JFK', 'LAX'): {'car_type': 'SUV', 'price_per_km': 0.25, 'distance': 4000},
    ('LHR', 'CDG'): {'car_type': 'Sedan', 'price_per_km': 0.20, 'distance': 450},
    ('AMS', 'BER'): {'car_type': 'Compact', 'price_per_km': 0.15, 'distance': 600},
    ('NRT', 'KIX'): {'car_type': 'Luxury', 'price_per_km': 0.30, 'distance': 500},
    ('DAL', 'JFK'): {'car_type': 'SUV', 'price_per_km': 0.25, 'distance': 2500},
    ('DAL', 'SEA'): {'car_type': 'Sedan', 'price_per_km': 0.22, 'distance': 3500},
    ('DAL', 'CDG'): {'car_type': 'Compact', 'price_per_km': 0.18, 'distance': 8000},
    ('DAL', 'DXB'): {'car_type': 'Luxury', 'price_per_km': 0.30, 'distance': 13000},
    ('CDG', 'LHR'): {'car_type': 'Sedan', 'price_per_km': 0.20, 'distance': 450},  # Paris to London
    ('BLR', 'MAA'): {'car_type': 'SUV', 'price_per_km': 0.25, 'distance': 350},  # Bangalore to Chennai
    ('BLR', 'DEL'): {'car_type': 'Luxury', 'price_per_km': 0.30, 'distance': 2000},  # Bangalore to New Delhi
    ('BLR', 'BOM'): {'car_type': 'Compact', 'price_per_km': 0.18, 'distance': 1000},
    ('JFK', 'NRT'): {'car_type': 'SUV', 'price_per_km': 0.25, 'distance': 40000}
    # Bangalore to Mumbai





    
}

def is_accessible_by_car(self):
    """
    Check if either origin or destination is a coastal city with an ocean.
    If either city is coastal, return False (not accessible by car).
    """
    if self.origin in self.ocean_cities or self.destination in self.ocean_cities:
        return False
    return True

def calculate_distance(self):
    """
    Return the travel distance based on predefined city distances.
    You can replace this with a more sophisticated distance calculation method.
    """
    city_distances = {
        ('New York', 'Los Angeles'): 4500,  # in kilometers
        ('London', 'Paris'): 340,  # in kilometers
        ('Berlin', 'Amsterdam'): 650,  # in kilometers
        ('Tokyo', 'Osaka'): 500,  # in kilometers
        # Add more city pairs as needed
    }
    return city_distances.get((self.origin, self.destination), 0)


def calculate_fuel_cost(self):
    """
    Calculate the fuel cost for the journey based on the distance and fuel price.
    """
    if not self.is_accessible_by_car():
        return "This route cannot be traveled by car due to ocean cities."

    distance = self.calculate_distance()
    if distance == 0:
        return "Distance data unavailable for this route."

    fuel_required = distance / self.fuel_efficiency  # Fuel required in liters
    total_cost = fuel_required * self.fuel_price  # Total cost of fuel
    return f"The estimated fuel cost for your trip is ${total_cost:.2f}"

    
# Function to calculate the total travel price
def calculate_travel_price(car_type, price_per_km, distance, traveler_count):
    # Calculate the total cost for the travel
    return price_per_km * distance * traveler_count




def car_travel_prompt():
    """
    A function to prompt the user for travel details and calculate the car travel cost.
    This includes inputs for origin, destination, fuel price, and fuel efficiency.
    """
    # Example manual data for cities with oceans
    ocean_cities = ["Miami", "Los Angeles", "New York", "Tokyo", "San Francisco"]  # Predefined coastal cities
    
    # Example user input (can be dynamic based on your actual input interface)
    origin = input("Enter your origin city: ")  # e.g., "London"
    destination = input("Enter your destination city: ")  # e.g., "Paris"
    fuel_price = float(input("Enter the current fuel price per liter (in USD): "))  # e.g., 1.5
    fuel_efficiency = float(input("Enter your car's fuel efficiency (in km per liter): "))  # e.g., 15
    
    # Create the car travel agent
    agent = CarTravelAgent(origin, destination, fuel_price, fuel_efficiency, ocean_cities)
    
    # Get and display the fuel cost result
    result = agent.calculate_fuel_cost()
    print(result)

def __init__(self, origin, destination, fuel_price, fuel_efficiency, ocean_cities):
    """
    Initialize the car travel agent with the given parameters.
    
    Parameters:
    - origin: Starting city for the trip.
    - destination: Destination city for the trip.
    - fuel_price: Price of fuel per liter (in USD).
    - fuel_efficiency: Fuel efficiency of the car (in km per liter).
    - ocean_cities: List of cities with oceans that cannot be traveled by car.
    """
    self.origin = origin
    self.destination = destination
    self.fuel_price = fuel_price
    self.fuel_efficiency = fuel_efficiency
    self.ocean_cities = ocean_cities
    
        
def fetch_flight_data(origin: str, destination: str, date: str,sender_email:str, travel_class: str = None,flight_data_limit=10)-> dict:
    flight_type = "Unknown"  # Initialize flight_type with a default value
    lowest_price_flight = None  # Initialize lowest_price_flight to avoid reference errors
    
    try:
        # Set travel class if provided; otherwise, default to ECONOMY
        travel_class = travel_class.upper() if travel_class else "ECONOMY"
        

        # Fetch flight offers
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=date,
            adults=1,
            travelClass=travel_class,
            returnDate=date
            # adults=adults,
            # children=children,
            # infants=infants     
            
        )

        flights = []

        for flight in response.data:
            # Get airline names
            airline_codes = flight.get("validatingAirlineCodes", [])
            airline_names = [get_airline_name(code) for code in airline_codes]
            city_name=flight.get("Validatingcityname",[])
            cityname=[get_city_name(subtype) for subtype in city_name]

            # Extract itinerary details
            itinerary = flight["itineraries"][0]["segments"]
            departure_time = itinerary[0]["departure"]["at"]
            arrival_time = itinerary[-1]["arrival"]["at"]

            # Calculate total time of flight in hours
            total_time_hours = calculate_total_time_in_hours(departure_time, arrival_time)


            # Calculate number of stops and stopover airports
            num_stops = len(itinerary) - 1
            stopover_airports = [segment["departure"]["iataCode"] for segment in itinerary[1:-1]]

             # flight type
            flight_type = direct_indirect_flight(num_stops)

            
            # Append flight details
            flights.append({
                "airline": ", ".join(airline_names),
                "price": float(flight["price"]["total"]),
                "currency": flight["price"]["currency"],
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "travel_class": travel_class,
                "total_time_hours": total_time_hours,
                "num_stops": num_stops,
                "flight_type":flight_type,
                "stopover_airports": stopover_airports
                
                
            })

            

            if len(flights) > flight_data_limit:
                break
                
        # Convert prices from EUR to USD if necessary
        flights = convert_prices_to_usd(flights)

        # Filter and sort flights
        cheap_flights = filter_and_sort_flights_by_price_and_duration(flights)
        cheapest_flight = cheap_flights[0] if cheap_flights else None

        #Cheap flight Only (Price)

        lowest_price_flight = find_lowest_price_flight(flights)

       


        return {
            "origin": origin,
            "destination": destination,
            "date": date,
            "flights": flights,
            "message": f"Found {len(flights)} flights from {origin} to {destination} on {date}.",
            "cheap_flight": cheapest_flight,
            "flight_type":flight_type,
            "lowest_price_flight": lowest_price_flight
        }

    except ResponseError as error:
        print(f"Error fetching flight data: {error}")
        return {
            "origin": origin,
            "destination": destination,
            "date": date,
            "flights": [],
            "message": f"An error occurred: {error}",
            "cheap_flight": None,
            "flight_type":flight_type,
            "lowest_price_flight": lowest_price_flight
        }


    
        
# Function to find the lowest price flight
def find_lowest_price_flight(flights):
    if flights:
        lowest_price_flight = min(flights, key=lambda x: x["price"])
        return lowest_price_flight
    return None






origin = email_data.origin
destination =  email_data.destination
max_price = 500
max_duration = 10
date = email_data.date
sender_email="2022ac05452@wilp.bits-pilani.ac.in"


# Define necessary inputs
origin = "JFK"  # Example origin airport code
destination = "LAX"  # Example destination airport code
date = "2025-04-01"  # Example travel date
sender_email = "user@example.com"  # Example email
travel_class = "ECONOMY"  # Default value


flight_data = fetch_flight_data(origin, destination, date, sender_email, travel_class)

if flight_data["lowest_price_flight"]:
    flight_price = flight_data["lowest_price_flight"]["price"]
else:
    flight_price = float('inf')  # Assign a high value if no flight is found

# Example car cost and passenger counts (ensure these values are set dynamically)
car_cost = 150.0  # Replace with actual car cost calculation
flight_passengers = 1  # Update based on user input
car_passengers = 1  # Update based on user input

# best_option, cost_difference, flight_price_per_person, car_price_per_person = compare_travel_options(
#     flight_price, flight_passengers, car_cost, car_passengers
# )


flight_details = fetch_flight_data(origin, destination, str(date), sender_email)
filtered_flights = [
                        
                flight for flight in flight_details["flights"]
                if flight["price"] <= 5000 and flight["total_time_hours"] <= 20
                ]

max_price=100
max_duration=10


# Define the AI agent using pydantic-ai
flight_agent = Agent(
    GroqModel(
        model_name="llama-3.3-70b-versatile",  # Replace with the AI model you are using
        api_key="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"  # Replace with your Groq API Key
    ),
    result_type=FlightSearchResult,
    system_prompt=(
        "You are a travel assistant that helps users find flight options." ,"You should consider the following constraints when filtering flight results: \n"
        "- **Flexible Dates:** Search for flights ±1 or ±2 days from the given date. \n"
        "- **Week Preference:** Ensure flights fall within the 2nd or 3rd week of the month if specified. \n"
        "- **Max Price & Duration:** Filter flights that exceed the max price or max duration constraints. \n"
        "airlineprice currency	departure_time	arrival_time	travel_class	total_time_hours	num_stops	flight_type	stopover_airports",
        f"""The above are the column headers kindly filter out with the limitations given below
        Actualdata:{filtered_flights}
        /nMax Price ($):{max_price} Max Duration (hrs):{max_duration} Origin Airport Code:{origin} Destination Airport Code:{destination} Departure Date:{date}"""
    ),
)



# Define the AI agent for fetching destination attractions
destination_attraction_agent = Agent(
    GroqModel(
        model_name="llama-3.3-70b-versatile",  # Replace with the AI model you are using
        api_key="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"  # Replace with your Groq API Key
    ),
    result_type=CategorizationResult,
    system_prompt=(
        "You are an AI travel assistant. Your task is to categorize the given list of experiences "
         "into 'Restaurant', 'Hotel', 'Attraction', or 'Other'."
    ),
)
# New hotel search agent
hotel_agent = Agent(
    GroqModel(
        model_name="llama-3.3-70b-versatile",
        api_key="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"
    ),
    result_type=HotelSearchResult,
    system_prompt=(
        "You are a travel assistant that helps users find the best hotel options."
        "Only return hotels where the price is greater than 0."
        "Provide the hotel name, price, and address."
    ),
)

car_agent = Agent(
    GroqModel(
        model_name="llama-3.3-70b-versatile",
        api_key="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"
    ),
    result_type=Carsearchresult,
    system_prompt=("""You are a virtual car rental assistant. You can calculate car travel costs based on distance, fuel price, and fuel efficiency. 
    Here’s how you should handle the calculations:
    
    1. **Check for Oceans:**
        - If either the origin or destination is a city with an ocean, ignore the route for car travel.
        - Use the following predefined list of coastal cities that have an ocean: ["Miami", "Los Angeles", "New York", "Tokyo", "San Francisco"].
    
    2. **Calculate Distance:**
        - Use predefined distances between city pairs. Example: 'New York' to 'Los Angeles' is 4500 km.
        - If the distance between the cities is not available, return a message: "Distance data unavailable for this route."
    
    3. **Calculate Fuel Requirements and Cost:**
        - The fuel required for the trip is calculated as: 
          `Fuel Required = Distance / Fuel Efficiency (in km per liter)`
        - The cost of the fuel is: 
          `Fuel Cost = Fuel Required * Fuel Price (per liter)`
        
    4. **Return Results:**
        - If the route cannot be traveled by car (due to ocean cities), return: "This route cannot be traveled by car due to ocean cities."
        - Otherwise, return the calculated fuel cost in the format: "The estimated fuel cost for your trip is $X.XX."
    
    Example Interaction:
    User: "Calculate the cost of traveling from London to Paris, fuel price is 1.5 USD per liter, and fuel efficiency is 15 km per liter."
    Response: "The estimated fuel cost for your trip is $34.00."
    
    Your task is to handle these kinds of user interactions with correct calculations and return the results.
    """
))


flight_passengers = 1
flight_price_per_person=1000
car_price_per_person=700
# Calculate flight price per person
# flight_price_per_person = lowest_price_flight["price"] / flight_passengers
# car_price_per_person = total_cost / traveler_count if traveler_count > 0 else 0

def compare_travel_options(flight_price_per_person: float, car_price_per_person: float):
    """
    Compares the lowest flight price per person with the car price per person and returns the best option.
    Args:
    - flight_price_per_person (float): The price per person for the flight.
    - car_price_per_person (float): The price per person for the car.

    Returns:
    - dict: A dictionary with the best option, cost difference, and recommendation.
    """
    if flight_price_per_person < car_price_per_person:
        best_option = "Flight"
        cost_difference = car_price_per_person - flight_price_per_person
    elif car_price_per_person < flight_price_per_person:
        best_option = "Car"
        cost_difference = flight_price_per_person - car_price_per_person
    else:
        best_option = "Equal cost"
        cost_difference = 0.0  # No difference if both options are equal

    # Prepare the result as a dictionary
    comparison_result = {
        "best_option": best_option,
        "cost_difference": cost_difference,
        "flight_price_per_person": flight_price_per_person,
        "car_price_per_person": car_price_per_person
    }

    return comparison_result

def main():
    st.set_page_config(layout="wide")  # Set layout to wide mode
    st.title("Travel Assistant")
    st.write("Welcome to the Flight,Hotel, Car and Destination activity Search App. Use the sliders to refine your options.")
     # Initialize session state to store email data
    if 'email_data' not in st.session_state:
        st.session_state.email_data = None


    # Add tab for email processing
    tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(["Fetch your Email", "Flight Search", "Hotel Finder", "Car Search", "Destination Experience","Travel Recommendation"])
    
    with tab1:
        st.header("Process Email for Travel Details")
        if st.button("Fetch Latest Email"):
            with st.spinner("Processing latest email..."):
                # Connect to email
                mail = connect_to_email(username, password)
                
                # Fetch latest email
                raw_email = fetch_latest_email(mail)
                if raw_email:
                    email_data, entities = parse_email(raw_email)
                    
                    # Store in session state
                    st.session_state.email_data = email_data
                    
                    # Display extracted information
                    st.success("Email processed successfully!")
                    st.write(f"Subject: {email_data.subject}")
                    st.write(f"Sender: {email_data.sender}")
                    st.write(f"Origin: {email_data.origin}")
                    st.write(f"Destination: {email_data.destination}")
                    st.write(f"Date: {email_data.date}")
                    
                    # Get city codes
                    origin_code = get_city_code(email_data.origin) if email_data.origin else None
                    destination_code = get_city_code(email_data.destination) if email_data.destination else None
                    
                    if origin_code and destination_code:
                        st.write(f"Origin Code: {origin_code}")
                        st.write(f"Destination Code: {destination_code}")
                    
                    mail.logout()
                else:
                    st.error("No emails found.")
    
    with tab2:
        st.header("Find Flights")
        col1, col2 = st.columns([1, 3])
        with col1:
            # Use email data if available
            default_origin = ""
            default_destination = ""
            default_date = None
            default_email = "example@gmail.com"
            
            if st.session_state.email_data:
                email_data = st.session_state.email_data
                if email_data.origin:
                    origin_code = get_city_code(email_data.origin)
                    default_origin = origin_code if origin_code else ""
                if email_data.destination:
                    destination_code = get_city_code(email_data.destination)
                    default_destination = destination_code if destination_code else ""
                if email_data.date:
                    try:
                        # Try to parse the date string into a datetime object
                        default_date = datetime.strptime(email_data.date, "%Y-%m-%d").date()
                    except ValueError:
                        try:
                            # Try alternative date formats
                            for fmt in ["%B %d, %Y", "%d %B %Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]:
                                try:
                                    default_date = datetime.strptime(email_data.date, fmt).date()
                                    break
                                except ValueError:
                                    continue
                        except:
                            # If all parsing attempts fail, use today's date
                            default_date = datetime.now().date()
                if email_data.sender:
                    default_email = email_data.sender
            
            max_price = st.slider("Max Price ($)", 50, 5000, 1000, 100)
            max_duration = st.slider("Max Duration (hrs)", 1, 20, 10, 1)
            origin = st.text_input("Origin Airport Code", default_origin)
            destination = st.text_input("Destination Airport Code", default_destination)
            date = st.date_input("Departure Date", default_date or datetime.now())
          #  sender_email = st.text_input("Email", default_email)

            st.subheader("Flexible Filter")
            flexible_option = st.radio(
                "Choose flexibility:", ["Exact Date", "±3 Days", "Weekend", "Specific Week"], 
                index=0)

            # Adjust date range based on selection
            if flexible_option == "±3 Days":
                date_range = (date - timedelta(days=3), date + timedelta(days=3))
                st.write(f"Searching flights from **{date_range[0]}** to **{date_range[1]}**")
        
            elif flexible_option == "Weekend":
                next_saturday = date + timedelta((5 - date.weekday()) % 7)
                next_sunday = next_saturday + timedelta(days=1)
                date_range = (next_saturday, next_sunday)
                st.write(f"Searching flights for the **weekend ({date_range[0]} - {date_range[1]})**")
        
            elif flexible_option == "Specific Week":
                # Let user select the month and week
                selected_month = st.selectbox("Select Month", [
                    "January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December"
                ], index=date.month - 1)
                
                selected_week = st.selectbox("Select Week", [
                    "First Week", "Second Week", "Third Week", "Fourth Week"
                ], index=0)
        
                # Convert month name to month number
                month_num = datetime.strptime(selected_month, "%B").month
                year = date.year  # Use current year
        
                # Get start date of the selected week
                first_day_of_month = datetime(year, month_num, 1)
                start_date = first_day_of_month + timedelta(days=(int(selected_week[0]) - 1) * 7)
                end_date = start_date + timedelta(days=6)
        
                date_range = (start_date.date(), end_date.date())
                st.write(f"Searching flights for **{selected_week} of {selected_month} ({date_range[0]} - {date_range[1]})**")
        
            else:
                date_range = (date, date)
                st.write(f"Searching flights for **{date}**")
            
            
    with col2:
        if st.button("Search Flights"):
        # Fetch Flight Data (Replace with actual API function)
            flight_details = fetch_flight_data(origin, destination, str(date), sender_email)

        # Ensure data is present
            if "flights" in flight_details and flight_details["flights"]:
            # Convert data to DataFrame
                flights_df = pd.DataFrame(flight_details["flights"])

            # Ensure correct data types
                flights_df["price"] = pd.to_numeric(flights_df["price"], errors="coerce")
                flights_df["total_time_hours"] = pd.to_numeric(flights_df["total_time_hours"], errors="coerce")

            # Apply Filtering
                filtered_flights = flights_df[
                    (flights_df["price"] <= max_price) & 
                    (flights_df["total_time_hours"] <= max_duration)
                ]

            # Display Results
                if not filtered_flights.empty:
                    st.write(f"### Filtered Flights (Max Price: {max_price}, Max Duration: {max_duration} hrs)")
                    st.table(filtered_flights)
                else:
                    st.warning("No flights match your criteria.")
            else:
                st.error("No flight data found. Please try different inputs.")
    
    with tab3:
        st.header("Find Hotels")
        
        # Define variables only used within tab3
        # Use the destination from email data if available
        tab3_default_destination_city = ""
        tab3_default_checkin_date = datetime.now()
        
        if st.session_state.email_data and st.session_state.email_data.destination:
            tab3_default_destination_city = st.session_state.email_data.destination
        
        # Use the date from email data for check-in if available
        if st.session_state.email_data and st.session_state.email_data.date:
            try:
                # Try to parse the date string into a datetime object
                tab3_default_checkin_date = datetime.strptime(st.session_state.email_data.date, "%Y-%m-%d")
            except ValueError:
                try:
                    # Try alternative date formats
                    for fmt in ["%B %d, %Y", "%d %B %Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]:
                        try:
                            tab3_default_checkin_date = datetime.strptime(st.session_state.email_data.date, fmt)
                            break
                        except ValueError:
                            continue
                except:
                    # If all parsing attempts fail, use today's date
                    tab3_default_checkin_date = datetime.now()
        
        # Hotel search interface - all variables with tab3_ prefix to scope them
        tab3_city_name = st.text_input("Enter City", tab3_default_destination_city, key="tab3_hotel_city")
        tab3_travelers = st.number_input("Travelers", 1, 10, 1, key="tab3_travelers")
        tab3_check_in = st.date_input("Check-in Date", tab3_default_checkin_date, key="tab3_check_in")
        tab3_check_out = st.date_input("Check-out Date", tab3_default_checkin_date + pd.Timedelta(days=5), key="tab3_check_out")
        tab3_min_price, tab3_max_price = st.slider("Price Range", 0, 1000, (0, 500), 50, key="tab3_price_range")
        tab3_board_type = st.selectbox("Board Type", ["All", "Full Board", "Half Board", "Bed & Breakfast", "Room Only", "All Inclusive"],  key="tab3_board_type")
        
        # Hotel search button only in tab3
        if st.button("Search Hotels", key="tab3_hotel_search_button"):
            with st.spinner("Searching for hotels..."):
                tab3_city_code = get_city_code(tab3_city_name)
                if tab3_city_code:
                    tab3_hotels = get_hotels(tab3_city_code, tab3_min_price, tab3_max_price)
                    if tab3_hotels:
                        st.success(f"Found {len(tab3_hotels)} hotels in {tab3_city_name}")
                        tab3_hotel_df = pd.DataFrame(tab3_hotels)
                        st.dataframe(tab3_hotel_df)
                        
                        # # Allow selecting a hotel to see available offers
                        # if not tab3_hotel_df.empty:
                        #     tab3_selected_hotel = st.selectbox(
                        #         "Select a hotel to view available offers:",
                        #         options=tab3_hotel_df["name"].tolist(),
                        #         index=0,
                        #         key="tab3_selected_hotel"
                        #     )
                            
                            # Get the hotel ID for the selected hotel
                       # tab3_selected_hotel_id = tab3_hotel_df[tab3_hotel_df["name"] == tab3_selected_hotel]["hotel_id"].iloc[0]
                            
                else:
                    st.error(f"Could not find city code for {tab3_city_name}. Please check the city name.")
    
    with tab4:
        st.header("Car Travel")
        tab4_default_origin_code = ""
        tab4_default_destination_code = ""
        tab4_default_start_date = datetime.now()
        # Use the origin from email data if available (assuming it's stored in origin field)
        if st.session_state.email_data and hasattr(st.session_state.email_data, 'origin'):
            tab4_default_origin_code = st.session_state.email_data.origin
        
        # Use the destination from email data if available
        if st.session_state.email_data and hasattr(st.session_state.email_data, 'destination'):
            tab4_default_destination_code = st.session_state.email_data.destination
        
        # Use the date from email data for start date if available
        if st.session_state.email_data and hasattr(st.session_state.email_data, 'date'):
            try:
                # Try to parse the date string into a datetime object
                tab4_default_start_date = datetime.strptime(st.session_state.email_data.date, "%Y-%m-%d")
            except ValueError:
                try:
                    # Try alternative date formats
                    for fmt in ["%B %d, %Y", "%d %B %Y", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"]:
                        try:
                            tab4_default_start_date = datetime.strptime(st.session_state.email_data.date, fmt)
                            break
                        except ValueError:
                            continue
                except:
                    # If all parsing attempts fail, use today's date
                    tab4_default_start_date = datetime.now()
        
        # Car rental filters inside tab4
        origin_code = st.selectbox("Select the origin airport code:", 
                                  ['JFK', 'LHR', 'AMS', 'NRT'], 
                                  index=(['JFK', 'LHR', 'AMS', 'NRT'].index(tab4_default_origin_code) if tab4_default_origin_code in ['JFK', 'LHR', 'AMS', 'NRT'] else 0),
                                  key="tab4_origin")
        
        destination_code = st.selectbox("Select the destination airport code:", 
                                       ['LAX', 'CDG', 'BER', 'KIX'],
                                       index=(['LAX', 'CDG', 'BER', 'KIX'].index(tab4_default_destination_code) if tab4_default_destination_code in ['LAX', 'CDG', 'BER', 'KIX'] else 0),
                                       key="tab4_destination")
        
        traveler_count = st.number_input("Enter the number of travelers:", 
                                        min_value=1, 
                                        max_value=10, 
                                        value=(st.session_state.email_data.passengers if hasattr(st.session_state.email_data, 'passengers') else 1),
                                        key="tab4_travelers")
        
        start_date = st.date_input("Select the start date:", 
                                  tab4_default_start_date,
                                  key="tab4_start_date")
        
        end_date = st.date_input("Select the end date:", 
                                tab4_default_start_date + pd.Timedelta(days=5),
                                key="tab4_end_date")
        
        # Define default values
        car_type = "N/A"
        price_per_km = 0
        distance = 0
        total_cost = 0
        
        # Add a search button
        if st.button("Search Car Rentals", key="tab4_car_search_button"):
            # Look up car data based on the origin and destination airport codes
            if (origin_code, destination_code) in car_data:
                car_info = car_data[(origin_code, destination_code)]
                car_type = car_info['car_type']
                price_per_km = car_info['price_per_km']
                distance = car_info['distance']
                
                # Calculate travel price
                total_cost = calculate_travel_price(car_type, price_per_km, distance, traveler_count)
                
                # Display the results
                st.success(f"Found car rental options for {origin_code} to {destination_code}")
                st.write(f"**Car Type**: {car_type}")
                st.write(f"**Distance**: {distance} km")
                st.write(f"**Price per km**: ${price_per_km}")
                st.write(f"**Total travel cost for {traveler_count} travelers**: ${total_cost:.2f}")
            else:
                st.warning("No travel data available for the entered airport codes.")
    
    
                
    with tab5:
         st.header(" Destination Activities")
         
         # Default city name from email data
         default_city = ""
         if st.session_state.email_data and st.session_state.email_data.destination:
             # Use destination from email as default
             default_city = st.session_state.email_data.destination
        
         city_name = st.text_input("Enter City Name (e.g., 'Paris', 'New York', 'Bangalore')", default_city)
        
         # If we have the destination name from email, display it
         if st.session_state.email_data and st.session_state.email_data.destination and not default_city:
             st.write(f"Using destination from email: {st.session_state.email_data.destination}")
    
         if st.button("Fetch Experiences"):
             with st.spinner("Fetching experiences..."):
                 city_name_str = str(city_name).strip()
                 if not city_name_str:
                     st.error("Please enter a valid city name.")
                     return
        
            # Debugging the content before passing to API
             st.write(f"User input city name (cleaned): '{city_name_str}'")
        
             coordinates = get_coordinates_from_city_name(city_name_str)
        
             if coordinates:
                 st.success(f"Found coordinates for {city_name_str} directly")
             else:
                 city_code = get_city_code_from_name(city_name_str)
                 if city_code:
                     coordinates = get_city_coordinates_from_amadeus(city_code)
                     if coordinates:
                         st.success(f"Found coordinates for {city_name_str} using city code {city_code}")

             if not coordinates:
                 st.error(f"Could not find coordinates for '{city_name_str}'. Please check and try again.")
                 return

             latitude, longitude = coordinates
 


             data_cor = fetch_destination_experiences(latitude, longitude)
             response = destination_attraction_agent.run_sync(str(data_cor[100:110]))
             
              # **Extract and Separate Categories**
             if response and response.data:
                 categorization_result_dict = response.data.model_dump()

                # Extract experiences into separate variables
                 restaurants = categorization_result_dict.get("categorized_experiences", {}).get("Restaurant", [])
                 hotels = categorization_result_dict.get("categorized_experiences", {}).get("Hotel", [])
                 attractions = categorization_result_dict.get("categorized_experiences", {}).get("Attraction", [])
                 other_experiences = categorization_result_dict.get("categorized_experiences", {}).get("Other", [])

                # **Streamlit Tabs for Categories**
                 tab1, tab2, tab3, tab4 = st.tabs(["Restaurants", "Hotels", "Attractions", "Other"])

                # **Restaurant Tab**
                 with tab1:
                     if not restaurants:
                         st.warning("No restaurants found!")
                     else:
                         for r in restaurants:
                             st.subheader(r["name"])
                             st.write(r["description"])
                             if r.get("image_url"):
                                 st.image(str(r["image_url"]))
                             if r.get("booking_link"):
                                 st.markdown(f"[Book Here]({str(r['booking_link'])})", unsafe_allow_html=True)
                             st.divider()

                # **Hotel Tab**
                 with tab2:
                     if not hotels:
                         st.warning("No hotels found!")
                     else:
                         for h in hotels:
                             st.subheader(h["name"])
                             st.write(h["description"])
                             if h.get("image_url"):
                                 st.image(str(h["image_url"]))
                             if h.get("booking_link"):
                                 st.markdown(f"[Book Here]({str(h['booking_link'])})", unsafe_allow_html=True)
                             st.divider()

                 # **Attraction Tab**
                 with tab3:
                     if not attractions:
                         st.warning("No attractions found!")
                     else:
                         for a in attractions:
                             st.subheader(a["name"])
                             st.write(a["description"])
                             if a.get("image_url"):
                                 st.image(str(a["image_url"]))
                             if a.get("booking_link"):
                                 st.markdown(f"[Book Here]({str(a['booking_link'])})", unsafe_allow_html=True)
                             st.divider()

                 # **Other Experiences Tab**
                 with tab4:
                     if not other_experiences:
                         st.warning("No other experiences found!")
                     else:
                         for o in other_experiences:
                             st.subheader(o["name"])
                             st.write(o["description"])
                             if o.get("image_url"):
                                 st.image(str(o["image_url"]))
                             if o.get("booking_link"):
                                 st.markdown(f"[Book Here]({str(o['booking_link'])})", unsafe_allow_html=True)
                             st.divider()

             else:
                 st.error("AI agent returned no categorization.")


             


    with tab6:
         # Create a display in Streamlit
        st.header("Travel Options Comparison")
        
       # Call the compare_travel_options function to get the best option and cost difference
        comparison_result = compare_travel_options(flight_price_per_person, car_price_per_person)
    
        # Display the per-person costs
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Flight Cost Per Person", f"${comparison_result['flight_price_per_person']:.2f}")
        with col2:
            st.metric("Car Cost Per Person", f"${comparison_result['car_price_per_person']:.2f}")
    
        # Display the recommendation
        st.markdown("---")
        if comparison_result["best_option"] == "Equal cost":
            st.info("Both options cost the same per person. Consider other factors like convenience, travel time, or preferences.")
        else:
            st.success(f"**Best Option: {comparison_result['best_option']}** (Saves ${comparison_result['cost_difference']:.2f} per person)")
    
        # Additional context (travel options data)
        travel_options = {
            "Flight": {
                "price_per_person": comparison_result['flight_price_per_person'],
                "total_price": flight_price,
                "passengers": flight_passengers
            },
            "Car": {
                "price_per_person": comparison_result['car_price_per_person'],
                "total_price": car_cost,
                "passengers": car_passengers
            }
        }
    
        # Display details in a dataframe
        comparison_df = {
            "Option": ["Flight", "Car"],
            "Total Cost": [f"${flight_price:.2f}", f"${car_cost:.2f}"],
            "Passengers": [flight_passengers, car_passengers],
            "Cost Per Person": [f"${comparison_result['flight_price_per_person']:.2f}", f"${comparison_result['car_price_per_person']:.2f}"]
        }
        st.dataframe(comparison_df) # Call the compare_travel_options function to get the best option and cost difference
        comparison_result = compare_travel_options(flight_price_per_person, car_price_per_person)
    
       
            
   
if __name__ == "__main__":
    main()


        # Now, use the extracted origin, destination, and date to fetch flight data
if email_data.origin and email_data.destination:
    origin_code = get_city_code(email_data.origin)
    destination_code = get_city_code(email_data.destination)

    print(f"Origin City Code: {origin_code}")
    print(f"Destination City Code: {destination_code}")

    
    
flights = fetch_flight_data(origin_code,destination_code,email_data.date,email_data.sender)



    

# Print the results
print(f"Flight Search Results:")
print(f"  Origin Airport: {flights['origin']}")
print(f"  Destination Airport: {flights['destination']}")
print(f"  Message: {flights['message']}")

print("\nAvailable Flights:")
for flight in flights["flights"]:
    print(f"  - Airline: {flight['airline']}, Price: ${flight['price']} {flight['currency']},Origin:{flights['origin']},Destination:{flights['destination']}, TravelClass: {flight['travel_class']}, Flightype: {flight['flight_type']},Departure: {flight['departure_time']}, TotalHours: {flight['total_time_hours']} hours,Stops: {flight['num_stops']}")

# Print the cheapest flight if available
if flights["cheap_flight"]:
    print("\nCheap and Less Flying Hours Flight:")
    
    cheap = flights["cheap_flight"]
    print(f"  - Airline: {cheap['airline']}, Price: ${cheap['price']} {cheap['currency']}, "
          f"Departure: {cheap['departure_time']},TravelClass:{cheap['travel_class']}, Stops: {cheap['num_stops']},Traveltype: {cheap['flight_type']}, TotalHours: {cheap['total_time_hours']} hours")

if flights["lowest_price_flight"]:
    lowest = flights["lowest_price_flight"]
    print("\nLowest Price Flight:")
    print(f"  - Airline: {lowest['airline']}, Price: ${lowest['price']} {lowest['currency']}, Departure: {lowest['departure_time']},TravelClass:{lowest['travel_class']},flighttype: {lowest['flight_type']} , Stops: {lowest['num_stops']}, TotalHours: {lowest['total_time_hours']} hours")
else:
    print("\nNo flights available.")



# result = flight_agent.run_sync(f"""
# The following are the column headers in the filtered_flights DataFrame:
# airline, price_currency, departure_time, arrival_time, travel_class, total_time_hours, num_stops, flight_type, and stopover_airports.

# Please filter the data based on these conditions:
# - Maximum Price: ${max_price}
# - Maximum Duration: {max_duration} hours
# - Origin: {origin}
# - Destination: {destination}
# - Departure Date: {date}

# Filtered Flights:{filtered_flights}
# """)

nest_asyncio.apply()

result=flight_agent.run_sync(f"""The following are the column headers in the filtered_flights DataFrame: airlineprice_currency, departure_time, arrival_time, travel_class, total_time_hours, num_stops, flight_type, and stopover_airports.

Please filter the data based on the given conditions:

Maximum Price: ${max_price}
Maximum Duration: {max_duration} hours
Origin Airport Code: {origin}
Destination Airport Code: {destination}
Departure Date: {date}
Actual Data:
{filtered_flights}

Return only the flights that meet these criteria""")




def send_email_with_flight_details(flight_data):
    """
    Send an email with the cheapest flight details to the extracted sender's email address.
    
    Parameters:
    flight_data (dict): The flight data dictionary containing flight information
    """
    # Get the receiver email from the email_data
    receiver_email = email_data.sender
    
    if not receiver_email:
        print("Error: No sender email address found in the email data.")
        return False
    
    # Email configuration
    sender_email = "airagenthelper@gmail.com"  # Replace with your email
    password = "nain bmym cfvp syjm"  # Replace with your app password

  
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"Your Cheapest Flight from {flight_data['origin']} to {flight_data['destination']}"
    
    # Get cheapest flight details
    cheapest_flight = flight_data.get('cheap_flight') or flight_data.get('lowest_price_flight')
    
    if not cheapest_flight:
        body = f"Sorry, no flights were found for your search from {flight_data['origin']} to {flight_data['destination']} on {flight_data['date']}. For more options visit http://localhost:8501/"
    else:
        # Format departure and arrival times
        departure_time = datetime.strptime(cheapest_flight['departure_time'], "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y at %I:%M %p")
        arrival_time = datetime.strptime(cheapest_flight['arrival_time'], "%Y-%m-%dT%H:%M:%S").strftime("%B %d, %Y at %I:%M %p")
        
        body = f"""
        Hello,
        
        Here is the cheapest flight we found for your trip:
        
        Route: {flight_data['origin']} to {flight_data['destination']}
        Date: {flight_data['date']}
        Airline: {cheapest_flight['airline']}
        Flight Type: {cheapest_flight['flight_type']}
        Price: ${cheapest_flight['price']} {cheapest_flight['currency']}
        Departure: {departure_time}
        Arrival: {arrival_time}
        Travel Class: {cheapest_flight['travel_class']}
        Total Flight Time: {cheapest_flight['total_time_hours']} hours
        Number of Stops: {cheapest_flight['num_stops']}
        
        Thank you for using our travel search service! For more options visit http://localhost:8501/
        
        Best regards,
        
        """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Send email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"Email with flight details successfully sent to {receiver_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Add this to the end of your main code to send the email with the cheapest flight details
if __name__ == "__main__":
    # After fetching and processing the email data and flight information
    if email_data and flights:
        # Make sure we have a sender email address
        if email_data.sender:
            print(f"Sending flight details to: {email_data.sender}")
            # Send the email with flight details
            send_email_with_flight_details(flights)
        else:
            print("Error: No sender email address found. Cannot send flight details.")
