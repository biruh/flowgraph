(ns flowgraph.forward
  (:require
    [flowgraph.core :as fg]
    [clojure.core.matrix.random :as r]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.operators :as ma]
    [clojure.core.async :as a :refer [>! <! >!! <!! go go-loop chan buffer close! thread alts! alts!! timeout]]
    ))

(import '[java.util Random])

(m/set-current-implementation :vectorz)  ;; use Vectorz as default matrix

(defn sigmoid-activation [weight bias input]
  (m/logistic (m/add (m/mmul weight input) bias)))

(defn sigmoid-back [output target weight bias learning_rate]
  (let [diff   (m/mul (m/sub  target output) 2)
        deriv  (m/mul output (m/sub 1.0 output))
        gderiv (m/mul diff deriv)
        g_w    (m/mmul (m/transpose weight) gderiv)
        new_w  (m/add weight (m/mul learning_rate g_w))
        new_b  (m/add bias   (m/mul learning_rate gderiv))
        sumdif (reduce + diff)]
    {:weight new_w
     :bias new_b
     :difference sumdif}))

(defn sigmoid-layer [initial-options input-chan]
  (let [out-chan (chan)
        maint-chan (chan)]
    (go
      (let [firstinput (<! input-chan)]
        (loop  [w (:weight initial-options)
                b (:bias initial-options)
                l (:learning_rate initial-options)
                i firstinput]
               (let [output (sigmoid-activation w b i)]
                 (>! out-chan output)
                 (let [ni (<! input-chan)
                       nw (sigmoid-back output ni w b l)]
                   (>! maint-chan nw)
               (recur (:weight nw) (:bias nw) l ni))))))
    {:out out-chan :maint maint-chan}))





(defn mapNum2Vec [n]
  (map (fn [x]
         (if  (= n x) 1 0)) (range 9)))

(defn mapVec2Num [v]
  (reduce + (map-indexed (fn [idx x]
         (if (zero? x) 0 idx)) v)))

(defn print-progress [c]
  (go (while true
        (println (<! c)))))

(defn test-l []
 (let [data (flatten (map (fn [x] [(m/array (mapNum2Vec x)) (m/array (mapNum2Vec (* x 2)))])
                          (repeatedly 100 #(rand-int 5))))
       input-chan (chan)
       initial-weights (m/array (repeatedly 9 #(r/sample-normal 9)))
       initial-bias (r/sample-normal 9)
       l  (sigmoid-layer {:weight initial-weights
                          :bias initial-bias
                          :learning_rate 0.1} input-chan)]
   (print-progress (:maint l))
   (println "Start")
   (go
       (doseq [i data]
         (>!! input-chan i)
         (println (<! (:out l)))
         ))))
