import numpy as np
from json import loads, dumps, dump
import os, sys

num_json_entries = 10000
debug_mode = "-d" in sys.argv
outdir = "./json_maps/"

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

# Generate a json lines file "dog_info.jsonl" containing 5000 lines, "each of which is a json object containing 3 keys: name, "color, "and breed. Each value should be randomly selected from a list of values, "where name is a random 5 character string, "color is either blue or green, "and breed is labrador or pitbull.

def gen_dog_info(num_entries, desired_props, json_file_name):
    prop_lists = []
    for prop in desired_props:
        prop_lists.append(np.random.choice(dog_info[prop], num_entries))

    if debug_mode: print(prop_lists)

    dog_info_lines = []
    for i in range(num_entries):
        curr_dict = {}
        for k, elem in enumerate(prop_lists):
            curr_dict[desired_props[k]] = str(elem[i])
        dog_info_lines.append(curr_dict)
    
    if debug_mode: print(dog_info_lines[0])
    json_lines = [dumps(l) for l in dog_info_lines]
    json_str = "\n".join(json_lines)
    with open(json_file_name, "w", encoding='utf-8') as f:
        f.write(json_str)

dog_info = {
    "name": ["Abbott", "Abby", "Abe", "Ace", "Addie", "Aero", "Aiden", "AJ", "Albert", "Alden", "Alex", "Alexis", "Alfie", "Alice", "Allie", "Alvin", "Alyssa", "Amber", "Amos", "Andy", "Angel", "Angus", "Anna", "Annie", "Apollo", "Archie", "Ariel", "Aries", "Artie", "Ash", "Ashley", "Aspen", "Athena", "Austin", "Autumn", "Ava", "Avery", "Axel", "Baby", "Bailey", "Bandit", "Barkley", "Barney", "Baron", "Basil", "Baxter", "Bean", "Bear", "Beau", "Bella", "Belle", "Benji", "Benny", "Bentley", "Betsy", "Betty", "Bianca", "Billy", "Bingo", "Birdie", "Biscuit", "Blake", "Blaze", "Blondie", "Blossom", "Blue", "Bo", "Bonnie", "Boomer", "Brady", "Brandy", "Brody", "Brooklyn", "Brownie", "Bruce", "Bruno", "Brutus", "Bubba", "Buck", "Buddy", "Buffy", "Buster", "Butch", "Buzz", "Cain", "Callie", "Camilla", "Candy", "Captain", "Carla", "Carly", "Carmela", "Carter", "Casey", "Cash", "Casper", "Cassie", "Champ", "Chance", "Chanel", "Charlie", "Chase", "Chester", "Chewy", "Chico", "Chief", "Chip", "Chloe", "Cinnamon", "CJ", "Cleo", "Clifford", "Clyde", "Coco", "Cody", "Colby", "Cookie", "Cooper", "Copper", "Cricket", "Daisy", "Dakota", "Damien", "Dana", "Dane", "Dante", "Daphne", "Darla", "Darlene", "Delia", "Delilah", "Denver", "Destiny", "Dexter", "Diamond", "Diego", "Diesel", "Diva", "Dixie", "Dodge", "Dolly", "Drew", "Duchess", "Duke", "Dylan", "Eddie", "Eden", "Edie", "Eli", "Ella", "Ellie", "Elmer", "Elsa", "Emma", "Emmett", "Emmy", "Eva", "Evan", "Faith", "Fanny", "Felix", "Fern", "Finn", "Fiona", "Fisher", "Flash", "Foxy", "Frankie", "Freddy", "Fritz", "Gabby", "Gage", "Gemma", "George", "Georgia", "Gia", "Gidget", "Gigi", "Ginger", "Gizmo", "Goldie", "Goose", "Gordie", "Grace", "Gracie", "Greta", "Griffin", "Gunner", "Gus", "Gypsy", "Hailey", "Hank", "Hannah", "Harley", "Harper", "Harvey", "Hawkeye", "Hazel", "Heidi", "Henry", "Hershey", "Holly", "Honey", "Hope", "Hoss", "Huck", "Hunter", "Ibby", "Iggy", "Inez", "Isabella", "Ivan", "Ivy", "Izzy", "Jack", "Jackie", "Jackson", "Jada", "Jade", "Jake", "Jasmine", "Jasper", "Jax", "Jenna", "Jersey", "Jesse", "Jessie", "Jill", "Joey", "Johnny", "Josie", "Judge", "Julia", "Juliet", "Juno", "Kali", "Kallie", "Kane", "Karma", "Kate", "Katie", "Kayla", "Kelsey", "Khloe", "Kiki", "King", "Kira", "Kobe", "Koda", "Koko", "Kona", "Lacy", "Lady", "Layla", "Leia", "Lena", "Lenny", "Leo", "Leroy", "Levi", "Lewis", "Lexi", "Libby", "Liberty", "Lily", "Lizzy", "Logan", "Loki", "Lola", "London", "Louie", "Lucky", "Luke", "Lulu", "Luna", "Mabel", "Mackenzie", "Macy", "Maddie", "Madison", "Maggie", "Maisy", "Mandy", "Marley", "Marty", "Matilda", "Mattie", "Maverick", "Max", "Maximus", "Maya", "Mia", "Mickey", "Mika", "Mila", "Miles", "Miley", "Millie", "Milo", "Mimi", "Minnie", "Missy", "Misty", "Mitzi", "Mocha", "Moe", "Molly", "Moose", "Morgan", "Morris", "Moxie", "Muffin", "Murphy", "Mya", "Nala", "Ned", "Nell", "Nellie", "Nelson", "Nero", "Nico", "Nikki", "Nina", "Noah", "Noel", "Nola", "Nori", "Norm", "Oakley", "Odie", "Odin", "Olive", "Oliver", "Olivia", "Ollie", "Oreo", "Oscar", "Otis", "Otto", "Ozzy", "Pablo", "Paisley", "Pandora", "Paris", "Parker", "Peaches", "Peanut", "Pearl", "Pebbles", "Penny", "Pepper", "Petey", "Phoebe", "Piper", "Pippa", "Pixie", "Polly", "Poppy", "Porter", "Precious", "Prince", "Princess", "Priscilla", "Quincy", "Radar", "Ralph", "Rambo", "Ranger", "Rascal", "Raven", "Rebel", "Reese", "Reggie", "Remy", "Rex", "Ricky", "Rider", "Riley", "Ringo", "Rocco", "Rockwell", "Rocky", "Romeo", "Rosco", "Rose", "Rosie", "Roxy", "Ruby", "Rudy", "Rufus", "Rusty", "Sadie", "Sage", "Sally", "Sam", "Samantha", "Sammie", "Sammy", "Samson", "Sandy", "Sarge", "Sasha", "Sassy", "Savannah", "Sawyer", "Scarlet", "Scooby", "Scooter", "Scout", "Scrappy", "Shadow", "Shamus", "Sheba", "Shelby", "Shiloh", "Sierra", "Simba", "Simon", "Sissy", "Sky", "Smokey", "Smoky", "Snickers", "Snoopy", "Sophia", "Sophie", "Sparky", "Spencer", "Spike", "Spot", "Stanley", "Star", "Stella", "Stewie", "Storm", "Sugar", "Suki", "Summer", "Sunny", "Sweetie", "Sydney", "Taco", "Tank", "Tasha", "Taz", "Teddy", "Tesla", "Tessa", "Theo", "Thor", "Tilly", "Titus", "TJ", "Toby", "Tootsie", "Trapper", "Tripp", "Trixie", "Tucker", "Tyler", "Tyson", "Vince", "Vinnie", "Violet", "Wally", "Walter", "Watson", "Willow", "Willy", "Winnie", "Winston", "Woody", "Wrigley", "Wyatt", "Xena", "Yogi", "Yoshi", "Yukon", "Zane", "Zelda", "Zeus", "Ziggy", "Zoe"],
    "breed": ["Afador", "Affenhuahua", "Affenpinscher", "Afghan Hound", "Airedale Terrier", "Akbash", "Akita", "Akita Chow", "Akita Pit", "Akita Shepherd", "Alaskan Klee Kai", "Alaskan Malamute", "American Bulldog", "American English Coonhound", "American Eskimo Dog", "American Foxhound", "American Hairless Terrier", "American Leopard Hound", "American Pit Bull Terrier", "American Pugabull", "American Staffordshire Terrier", "American Water Spaniel", "Anatolian Shepherd Dog", "Appenzeller Sennenhunde", "Auggie", "Aussiedoodle", "Aussiepom", "Australian Cattle Dog", "Australian Kelpie", "Australian Retriever", "Australian Shepherd", "Australian Shepherd Husky", "Australian Shepherd Lab Mix", "Australian Shepherd Pit Bull Mix", "Australian Stumpy Tail Cattle Dog", "Australian Terrier", "Azawakh", "Barbet", "Basenji", "Bassador", "Basset Fauve de Bretagne", "Basset Hound", "Basset Retriever", "Bavarian Mountain Scent Hound", "Beabull", "Beagle", "Beaglier", "Bearded Collie", "Bedlington Terrier", "Belgian Malinois", "Belgian Sheepdog", "Belgian Tervuren", "Bergamasco Sheepdog", "Berger Picard", "Bernedoodle", "Bernese Mountain Dog", "Bichon Frise", "Biewer Terrier", "Black and Tan Coonhound", "Black Mouth Cur", "Black Russian Terrier", "Bloodhound", "Blue Lacy", "Bluetick Coonhound", "Bocker", "Boerboel", "Boglen Terrier", "Bohemian Shepherd", "Bolognese", "Borador", "Border Collie", "Border Sheepdog", "Border Terrier", "Bordoodle", "Borzoi", "BoShih", "Bossie", "Boston Boxer", "Boston Terrier", "Boston Terrier Pekingese Mix", "Bouvier des Flandres", "Boxador", "Boxer", "Boxerdoodle", "Boxmatian", "Boxweiler", "Boykin Spaniel", "Bracco Italiano", "Braque du Bourbonnais", "Briard", "Brittany", "Broholmer", "Brussels Griffon", "Bugg", "Bull-Pei", "Bull Terrier", "Bullador", "Bullboxer Pit", "Bulldog", "Bullmastiff", "Bullmatian", "Cairn Terrier", "Canaan Dog", "Cane Corso", "Cardigan Welsh Corgi", "Carolina Dog", "Catahoula Bulldog", "Catahoula Leopard Dog", "Caucasian Shepherd Dog", "Cav-a-Jack", "Cavachon", "Cavador", "Cavalier King Charles Spaniel", "Cavapoo", "Central Asian Shepherd Dog", "Cesky Terrier", "Chabrador", "Cheagle", "Chesapeake Bay Retriever", "Chi Chi", "Chi-Poo", "Chigi", "Chihuahua", "Chilier", "Chinese Crested", "Chinese Shar-Pei", "Chinook", "Chion", "Chipin", "Chiweenie", "Chorkie", "Chow Chow", "Chow Shepherd", "Chug", "Chusky", "Cirneco dell’Etna", "Clumber Spaniel", "Cockalier", "Cockapoo", "Cocker Spaniel", "Collie", "Corgi Inu", "Corgidor", "Corman Shepherd", "Coton de Tulear", "Croatian Sheepdog", "Curly-Coated Retriever", "Dachsador", "Dachshund", "Dalmatian", "Dandie Dinmont Terrier", "Daniff", "Deutscher Wachtelhund", "Doberdor", "Doberman Pinscher", "Docker", "Dogo Argentino", "Dogue de Bordeaux", "Dorgi", "Dorkie", "Doxiepoo", "Doxle", "Drentsche Patrijshond", "Drever", "Dutch Shepherd", "English Cocker Spaniel", "English Foxhound", "English Setter", "English Springer Spaniel", "English Toy Spaniel", "Entlebucher Mountain Dog", "Estrela Mountain Dog", "Eurasier", "Field Spaniel", "Fila Brasileiro", "Finnish Lapphund", "Finnish Spitz", "Flat-Coated Retriever", "Fox Terrier", "French Bulldog", "French Bullhuahua", "French Spaniel", "Frenchton", "Frengle", "German Longhaired Pointer", "German Pinscher", "German Shepherd Dog", "German Shepherd Pit Bull", "German Shepherd Rottweiler Mix", "German Sheprador", "German Shorthaired Pointer", "German Spitz", "German Wirehaired Pointer", "Giant Schnauzer", "Glen of Imaal Terrier", "Goberian", "Goldador", "Golden Cocker Retriever", "Golden Mountain Dog", "Golden Retriever", "Golden Retriever Corgi", "Golden Shepherd", "Goldendoodle", "Gollie", "Gordon Setter", "Great Dane", "Great Pyrenees", "Greater Swiss Mountain Dog", "Greyador", "Greyhound", "Hamiltonstovare", "Hanoverian Scenthound", "Harrier", "Havanese", "Hokkaido", "Horgi", "Huskita", "Huskydoodle", "Ibizan Hound", "Icelandic Sheepdog", "Irish Red and White Setter", "Irish Setter", "Irish Terrier", "Irish Water Spaniel", "Irish Wolfhound", "Italian Greyhound", "Jack-A-Poo", "Jack Chi", "Jack Russell Terrier", "Jackshund", "Japanese Chin", "Japanese Spitz", "Korean Jindo Dog", "Karelian Bear Dog", "Keeshond", "Kerry Blue Terrier", "King Shepherd", "Komondor", "Kooikerhondje", "Kuvasz", "Kyi-Leo", "Lab Pointer", "Labernese", "Labmaraner", "Labrabull", "Labradane", "Labradoodle", "Labrador Retriever", "Labrastaff", "Labsky", "Lagotto Romagnolo", "Lakeland Terrier", "Lancashire Heeler", "Leonberger", "Lhasa Apso", "Lhasapoo", "Lowchen", "Maltese", "Maltese Shih Tzu", "Maltipoo", "Manchester Terrier", "Mastador", "Mastiff", "Miniature Pinscher", "Miniature Schnauzer", "Morkie", "Mudi", "Mutt", "Neapolitan Mastiff", "Newfoundland", "Norfolk Terrier", "Norwegian Buhund", "Norwegian Elkhound", "Norwegian Lundehund", "Norwich Terrier", "Nova Scotia Duck Tolling Retriever", "Old English Sheepdog", "Otterhound", "Papillon", "Papipoo", "Peekapoo", "Pekingese", "Pembroke Welsh Corgi", "Petit Basset Griffon Vendéen", "Pharaoh Hound", "Pitsky", "Plott", "Pocket Beagle", "Pointer", "Polish Lowland Sheepdog", "Pomapoo", "Pomchi", "Pomeagle", "Pomeranian", "Pomsky", "Poochon", "Poodle", "Portuguese Podengo Pequeno", "Portuguese Water Dog", "Pug", "Pugalier", "Puggle", "Puginese", "Puli", "Pyredoodle", "Pyrenean Shepherd", "Rat Terrier", "Redbone Coonhound", "Rhodesian Ridgeback", "Rottador", "Rottle", "Rottweiler", "Saint Berdoodle", "Saint Bernard", "Saluki", "Samoyed", "Samusky", "Schipperke", "Schnoodle", "Scottish Deerhound", "Scottish Terrier", "Sealyham Terrier", "Sheepadoodle", "Shepsky", "Shetland Sheepdog", "Shiba Inu", "Shichon", "Shih-Poo", "Shih Tzu", "Shiloh Shepherd", "Shiranian", "Shollie", "Shorkie", "Siberian Husky", "Silken Windhound", "Silky Terrier", "Skye Terrier", "Sloughi", "Small Munsterlander Pointer", "Soft Coated Wheaten Terrier", "Spanish Mastiff", "Spinone Italiano", "Springador", "Stabyhoun", "Staffordshire Bull Terrier", "Standard Schnauzer", "Sussex Spaniel", "Swedish Vallhund", "Terripoo", "Texas Heeler", "Tibetan Mastiff", "Tibetan Spaniel", "Tibetan Terrier", "Toy Fox Terrier", "Treeing Tennessee Brindle", "Treeing Walker Coonhound", "Valley Bulldog", "Vizsla", "Weimaraner", "Welsh Springer Spaniel", "Welsh Terrier", "West Highland White Terrier", "Westiepoo", "Whippet", "Whoodle", "Wirehaired Pointing Griffon", "Xoloitzcuintli", "Yorkipoo", "Yorkshire Terrier"],
    "color": ["alizarin", "amaranth", "amber", "amethyst", "apricot", "aqua", "aquamarine", "asparagus", "auburn", "azure", "beige", "bistre", "black", "blue", "brass", "bronze", "brown", "buff", "burgundy", "cardinal", "carmine", "celadon", "cerise", "cerulean", "champagne", "charcoal", "chartreuse", "chestnut", "chocolate", "cinnabar", "cinnamon", "cobalt", "copper", "coral", "corn", "cornflower", "cream", "crimson", "cyan", "dandelion", "denim", "ecru", "emerald", "eggplant", "firebrick", "flax", "fuchsia", "gamboge", "gold", "goldenrod", "green", "grey", "harlequin", "heliotrope", "indigo", "ivory", "jade", "khaki", "lavender", "lemon", "lilac", "lime", "linen", "magenta", "magnolia", "malachite", "maroon", "mauve", "mustard", "myrtle", "ochre", "olive", "olivine", "orange", "orchid", "peach", "pear", "periwinkle", "persimmon", "pink", "platinum", "plum", "puce", "pumpkin", "purple", "razzmatazz", "red", "rose", "ruby", "russet", "rust", "saffron", "salmon", "sangria", "sapphire", "scarlet", "seashell", "sepia", "silver", "smalt", "tan", "tangerine", "taupe", "teal", "thistle", "tomato", "turquoise", "ultramarine", "vermilion", "violet", "viridian", "wheat", "white", "wisteria", "yellow", "zucchini"],
    "size": np.arange(10, 200, 5),
    "temperament": ["Afraid", "Aggravated", "Angry", "Anxious", "Ashamed", "Assertive", "Burdened", "Brave", "Calm", "Cautious", "Challenged", "Cheerful", "Cherished", "Comforted", "Contented", "Creative", "Curious", "Depressed", "Embarrassed", "Energized", "Envious", "Excited", "Furious", "Guilty", "Grumpy", "Happy", "Hopeful", "Humiliated", "Hurt", "Indifferent", "Insecure", "Irritated", "Lonely", "Loved", "Mad", "Optimistic", "Overwhelmed", "Panicked", "Peaceful", "Positive", "Pessimistic", "Prepared", "Proud", "Regretful", "Relieved", "Renewed", "Sad", "Shameful", "Skeptical", "Sorrowful", "Suicidal", "Worried"]
}

np.random.seed(42)

for i in range(1, 6):
    assert i <= len(list(dog_info.keys())), "Too many properties per dog"
    desired_props = np.random.choice(list(dog_info.keys()), i, replace=False)
    gen_dog_info(num_json_entries, desired_props, f"{outdir}/dog_info_propcount_{i}.jsonl")