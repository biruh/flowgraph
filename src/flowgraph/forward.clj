(ns flowgraph.forward
  (:require
    [flowgraph.core :as fg]
    [clojure.core.matrix.random :as r]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.operators :as ma]
    [clojure.core.async :as a :refer [>! <! >!! <!! go go-loop chan buffer close! thread alts! alts!! timeout]]
    ))

(use '(incanter core charts stats datasets))

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

(defn sigmoid-layer [options input-chan]
  (let [out-chan   (chan)
        maint-chan (chan)]
    (go-loop [w (:weight        options)
              b (:bias          options)
              l (:learning_rate options)
              i (<! input-chan)]
               (let [output (sigmoid-activation w b i)]
                 (>! out-chan output)
                 (let [ni (<! input-chan)
                       nw (sigmoid-back output ni w b l)]
                   (>! maint-chan (assoc nw :id (:id options)))
               (recur (:weight nw)
                      (:bias nw)
                      l
                      ni))))
    {:out   out-chan
     :maint maint-chan}))

(defn createLayer [values input-chan & {:keys [learning_rate] :or {learning_rate 0.1}}]
  (let [l   (sigmoid-layer (assoc values  :learning_rate learning_rate) input-chan)]
    l))

(defn createStack [items input-chan]
  (loop [idx 0
         prev-chan input-chan
         package []]
    (if (>= idx (count items))
      {:out prev-chan :maints package}
      (let [item  (nth items idx)
            l     (createLayer item prev-chan)
            out   (:out l)
            maint (:maint l)]
      (recur (inc idx) out (conj package maint))))))


(defn stack [items input-chan]
  (let [out-chan   (chan)
        maint-chan (chan)
        s          (createStack items input-chan)
        m          (a/merge (:maints s))]
    (go-loop []
         (>! out-chan (<! (:out s)))
         (>! maint-chan (<! m))
        (recur))
  {:out out-chan
   :maint maint-chan}))

(defn getNormalVector[size]
     (r/sample-normal size))

(defn getNormalMatrix [size]
     (m/array (repeatedly size #(r/sample-normal size))))

(defn init-Layer [id size]
  (let [initial-weights (getNormalMatrix size)
        initial-bias  (getNormalVector size)]
    {:id id :weight initial-weights :bias initial-bias}))


(defn mapNum2Vec [n]
  (map (fn [x]
         (if  (= n x) 1 0)) (range 9)))

(defn mapVec2Num [v]
  (reduce + (map-indexed (fn [idx x]
         (if (zero? x) 0 idx)) v)))

(defn print-progress [c]
  (go (while true
        (let [i (<! c)]
        (println (:id i) " = " (:difference i))))))

(defn generate-m2 []
  (flatten (map (fn [x] [(m/array (mapNum2Vec x)) (m/array (mapNum2Vec (* x 2)))])
                            (repeatedly 1000 #(rand-int 5)))))

(defn test-l []
 (let [data (generate-m2)
       input-chan (chan)
       l  (createLayer (init-Layer "test" 9) input-chan)]
   (print-progress (:maint l))
   (go
       (doseq [i data]
         (>!! input-chan i)
         (<! (:out l))))))

(defn test-m []
 (let [data (generate-m2)
       input-chan (chan)
       l  (stack [(init-Layer "l1" 9) (init-Layer "l2" 9)] input-chan)]
   (print-progress (:maint l))
   (go
       (doseq [i data]
         (>!! input-chan i)
         (<! (:out l))))))
