
#!/usr/bin/env python
# Implementation of collaborative filtering recommendation engine


from recommendation_data import dataset
from math import sqrt

def similarity_score(person1, person2):
	
	# Returns ratio Euclidean distance score of person1 and person2 

	both_viewed = {}		# To get both rated items by person1 and person2

	for item in dataset[person1]:
		if item in dataset[person2]:
			both_viewed[item] = 1

		# Conditions to check they both have an common rating items	
		if len(both_viewed) == 0:
			return 0

		# Finding Euclidean distance 
		sum_of_eclidean_distance = []	

		for item in dataset[person1]:
			if item in dataset[person2]:
				sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item],2))
		sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

		return 1/(1+sqrt(sum_of_eclidean_distance))

def pearson_correlation(person1, person2):

	# To get both rated items
	both_rated = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_rated[item] = 1

	number_of_ratings = len(both_rated)		
	
	# Checking for number of ratings in common
	if number_of_ratings == 0:
		return 0

	# Add up all the preferences of each user
	person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
	person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

	# Sum up the squares of preferences of each user
	person1_square_preferences_sum = sum([pow(dataset[person1][item],2) for item in both_rated])
	person2_square_preferences_sum = sum([pow(dataset[person2][item],2) for item in both_rated])

	# Sum up the product value of both preferences for each item
	product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

	# Calculate the pearson score
	numerator_value = product_sum_of_both_users - (person1_preferences_sum*person2_preferences_sum/number_of_ratings)
	denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum,2)/number_of_ratings) * (person2_square_preferences_sum -pow(person2_preferences_sum,2)/number_of_ratings))
	if denominator_value == 0:
		return 0
	else:
		r = numerator_value/denominator_value
		return r

# Aggiunta questa funzione
def euclidean_distance(person1, person2):
	# To get both rated items
	both_rated = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_rated[item] = 1

		# Conditions to check they both have an common rating items	
		if len(both_rated) == 0:
			return 0

		# Finding Euclidean distance 
		sum_of_eclidean_distance = []	

		for item in dataset[person1]:
			if item in dataset[person2]:
				sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item],2))
		sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

		return 1/(1+sqrt(sum_of_eclidean_distance))

def most_similar_users(person,number_of_users):
	# returns the number_of_users (similar persons) for a given specific person.
	scores = [(pearson_correlation(person,other_person),other_person) for other_person in dataset if  other_person != person ]
	
	# Sort the similar persons so that highest scores person will appear at the first
	scores.sort()
	scores.reverse()
	return scores[0:number_of_users]

def user_reommendations(person, similarity = pearson_correlation): # La funzione è stata modificata passando la funzione di similarità come argomento

	# Gets recommendations for a person by using a weighted average of every other user's rankings
	totals = {}
	simSums = {}
	rankings_list =[]
	for other in dataset:
		# don't compare me to myself
		if other == person:
			continue
		sim = similarity(person, other)
		#print(">>>>>>>",sim)

		# ignore scores of zero or lower
		if sim <=0: 
			continue
		for item in dataset[other]:

			# only score movies i haven't seen yet
			if item not in dataset[person] or dataset[person][item] == 0:

			# Similrity * score
				totals.setdefault(item,0)
				totals[item] += dataset[other][item]* sim
				# sum of similarities
				simSums.setdefault(item,0)
				simSums[item]+= sim

		# Create the normalized list

	rankings = [(total/simSums[item],item) for item,total in totals.items()]
	rankings.sort()
	rankings.reverse()
	# returns the recommended items
	recommendataions_list = {recommend_item: score for score, recommend_item in rankings}
	return recommendataions_list
		
def calculateSimilarItems(prefs,n=10):
        # Create a dictionary of items showing which other items they
        # are most similar to.
        result={}
        # Invert the preference matrix to be item-centric
        itemPrefs=transformPrefs(prefs)
        c=0
        for item in itemPrefs:
                # Status updates for large datasets
                c+=1
                if c%100==0: print("%d / %d" % (c,len(itemPrefs)))
                # Find the most similar items to this one
                scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
                result[item]=scores
        return result

# Aggiunta questa nuova funzione che sintetizza il codice da eseguire nel main
def describe(person, sim):
	for other in dataset:
		# don't compare me to myself
		if other == person:
			continue
		similarity = sim(person, other)
		if similarity <=0: 
			continue
		print(other, "\t-->\t", f"{similarity:.2f}")
		print("\t Movies \t\t Rating\t S.x")
		print("\t", "-" * 35)
		for key, value in dataset[other].items():
			if key not in dataset[person] or dataset[person][key] == 0:
				print("\t", key, "\t", value, "\t", f"{similarity * value:.2f}")
		print()
	print("Movies recommendations for Toby:")
	for key, value in user_reommendations('Toby', sim).items(): # default similarity is pearson_correlation
		print("\t", key, f"{value:.2f}")

from statistics import mean

# Stima delle valutazioni
def valutation_estimate(person):
	numerator = 0.0
	denominator = 0.0
	mean_ratings = {}
	estmated_valutations = {}

	#mean ratings pre-calculation
	for item in dataset:
		mean_ratings[item] = mean(dataset[item].values())

	#traverse all item rated by other users
	for other in dataset:
		#Exclude target user from valutation
		if other == person:
			continue
		
		for item in dataset[other]:
		# only score movies i haven't seen yet
			if item not in dataset[person] or dataset[person][item] == 0:
				
				sim = pearson_correlation(person, other)

				#apply valutation esitmate formula
				numerator += (sim * (dataset[other][item] - mean_ratings[other]))
				denominator += sim

				estmated_valutations[item] = numerator / denominator

	#add the mean rating of the target user
	for item in estmated_valutations:
		estmated_valutations[item] += mean_ratings[person]

	return estmated_valutations

person = 'Toby'

if __name__ == '__main__':
	print("Movie Recommendations with Pearson Correlation")
	describe(person, pearson_correlation)
	
	print("=" * 100)
	
	print("Movie Recommendations with Euclidean Distance")
	describe(person, euclidean_distance)

	print("=" * 100)

	print("Valutation Estimate")
	for key, value in valutation_estimate(person).items():
		print("\t", key, f"{value:.2f}")
