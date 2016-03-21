(ns flowgraph.forward
  (:require
    [flowgraph.core :as fg]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.operators :as ma]))

(import '[java.util Random])

(m/set-current-implementation :vectorz)  ;; use Vectorz as default matrix


(defn layer [input target weight learning_rate]
  (let [il     (first (m/shape input))
        bp     (/ il 2.0)
        output (m/emap (fn [x] (if (>= x bp) 1 0)) (m/mmul weight input))
        diff   (m/sub target output)
        sumdif  (reduce + diff)]
    (println "----")
    (println "target:" target)
    (println "output:" output)
    (println "diff:" diff)
    {:weight (m/array (map (fn [x] (m/add (m/slice weight x) (m/mul (m/slice diff x) learning_rate))) (range il)))
     :difference sumdif
     :output output}))


(defn iter [initial_weight data learning_rate]
  (loop [weight initial_weight
         diffs []
         idx   0]
    (if (>= idx (count data))
      {:weight weight
       :difference diffs}
      (let [dp (nth data idx)
            input (:value dp)
            label (:label dp)
            no (layer input label weight learning_rate)]
        (println "weight:" (:weight no))
      (recur (:weight no) (conj diffs (:difference no)) (inc idx))))))



(defn mapNum2Vec [n]
  (map (fn [x]
         (if  (= n x) 1 0)) (range 9)))

(defn mapVec2Num [v]
  (reduce + (map-indexed (fn [idx x]
         (if (zero? x) 0 idx)) v)))

(def sample-data (map (fn [x]
                        {:value (m/array (mapNum2Vec x))
                         :label (m/array (mapNum2Vec (* x 2)))}) (repeatedly 100 #(rand-int 5))))


(def initial-weights (m/array (repeatedly 9 #(m/array (repeatedly 9 rand)))))

(m/mmul initial-weights (m/array (mapNum2Vec 2)))

(def final-weights (iter initial-weights sample-data 0.1))


(layer (m/array (mapNum2Vec 2)) (m/array (mapNum2Vec 4)) (:weight final-weights) 0.01)
(println (:difference final-weights))

(def sample-data_2 [
                    {:value [0 1] :label [0 1]}
                    {:value [0 1] :label [0 1]}
                    {:value [1 0] :label [0 1]}
                    {:value [1 0] :label [0 1]}
                    {:value [1 1] :label [0 1]}
                    {:value [1 1] :label [0 1]}
                    {:value [0 0] :label [0 1]}
                    {:value [0 0] :label [0 1]}])
(def initial-weights (m/array (repeatedly 2 #(m/array (repeatedly 2 rand)))))
(def final-weights (iter initial-weights sample-data_2 0.1))
(layer (m/array [0 1]) (m/array [0 1]) (:weight final-weights) 0.1)
(println initial-weights)
(println final-weights)
