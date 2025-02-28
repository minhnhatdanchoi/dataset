//Below is an example Cypher script that loads your CSV (from GitHub) where the Languages column is already a dictionary (with keys as languages and values as levels). This script will:

-- Convert the dictionary string to valid JSON by replacing Python’s “None” with JSON’s null and converting single quotes to double quotes.
-- Convert that JSON string into a map.
-- Create an Employee node (using the Name column).
-- UNWIND the keys of the map to create (or match) Language nodes.
-- Create a relationship between the Employee and each Language.
-- If the language’s level is not null (or empty), create a Level node and a relationship from the Language to that Level.

LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/minhnhatdanchoi/dataset/7c43759d6577f6e681fbdc93fb016807c318a17b/name_lang.csv' AS row
// Convert the Languages dictionary string to valid JSON
WITH row, apoc.text.replace(row.Languages, "None", "null") AS temp
WITH row, apoc.text.replace(temp, "'", "\"") AS langJson
// Convert the JSON string to a map
WITH row, apoc.convert.fromJsonMap(langJson) AS langMap
MERGE (e:Employee {name: row.Name})
WITH row, langMap, e
UNWIND keys(langMap) AS lang
MERGE (l:Language {name: lang})
MERGE (e)-[:LANGUAGE]->(l)
FOREACH (lvl IN CASE WHEN langMap[lang] IS NOT NULL AND langMap[lang] <> "" THEN [langMap[lang]] ELSE [] END |
  MERGE (lev:Level {name: lvl})
  MERGE (l)-[:LEVEL]->(lev)
)
RETURN e, l

-- Explanation
-- Data Conversion:
-- • We first replace None with null and then replace single quotes with double quotes so that the dictionary string
-- (e.g. {'Vietnamese': None, 'English': 'C1 Advance'}) becomes valid JSON (e.g. {"Vietnamese": null, "English": "C1 Advance"}).
--
-- Mapping to a Map:
-- • apoc.convert.fromJsonMap(langJson) turns the JSON string into a map (dictionary)
-- where each key is a language and each value is its corresponding level (or null).
--
-- Creating Nodes and Relationships:
-- • An Employee node is created using the Name field.
-- • We UNWIND the keys of the map to iterate over each language. For each language,
-- we merge a Language node and create a LANGUAGE relationship from the employee to that language.
-- • The FOREACH with a CASE clause checks if the level exists (i.e. it’s not null or empty).
-- If so, it merges a Level node and creates a LEVEL relationship from the Language node to the Level node.
--
-- Conditional Logic:
-- • If a row’s Languages column is empty (or its map is empty), then UNWIND produces no rows—so no language or level nodes are created.
--
-- This script meets your requirements: if the level is absent (null) no Level node is created,
-- and if the language data is empty then only the Employee node is created without any relationships.
