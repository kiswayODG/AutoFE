# AutoFE
Observations globales sur les régressions
L’usage d’AutoFeat ou FeatureTools conduit quasiment toujours à une amélioration des performances, aussi bien en termes de R² (meilleure prédiction) que de RMSE (moindre erreur).

L’effet est particulièrement visible sur des datasets avec des target à var continue . 

AutoFeat se démarque légèrement par rapport à FeatureTools, avec des gains plus réguliers et significatifs.

Le modèle Ridge combiné à AutoFeat est souvent le plus performant en régression.

  AutoFE est bénéfique, surtout sur des jeux de données plus complexes ou bruités.
  AutoFeat > FeatureTools en général


Classification / Régression logistique
Contrairement à la régression, AutoFE n’apporte que peu, voire pas d’amélioration sur les tâches de classification.

Que ce soit pour la régression logistique, Naive Bayes ou Random Forest, les performances (accuracy / précision) restent stables, voire légèrement dégradées avec AutoFeat.
.