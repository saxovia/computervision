
terms = {'cat', 'dog', 'mammals', 'mouse', 'pet'};
doc_percentage = [5, 20, 2, 10, 60];
terms_prob = doc_percentage / 100;
N = 3;

idf = zeros(length(terms), 1);
for i = 1:length(terms)
    idf(i) = log2(1 / terms_prob(i));
end
for i = 1:length(terms)
    fprintf('IDF of %s: %.4f\n', terms{i}, idf(i));
end




D1 = 'Cat is a pet, dog is a pet, and mouse may be a pet too.';
D2 = 'Cat, dog and mouse are all mammals.';
D3 = 'Cat and dog get along well, but cat may eat a mouse.';
documents = {D1, D2, D3};

v1 = clean(D1);
v2 = clean(D2);
v3 = clean(D3);
tf_q = build_tf(clean('mouse cat pet mammals dog'));
tf_v1 = build_tf(v1); 
tf_v2 = build_tf(v2); 
tf_v3 = build_tf(v3); 
%disp('Cleaned Doc 1:');
%disp(v1);
disp('Term freq for Query:');
disp_map(tf_q);

disp('Term freq for Document 1:');
disp_map(tf_v1);

disp('Term freq for Document 2:');
disp_map(tf_v2);

disp('Term freq for Document 3:');
disp_map(tf_v3);




tf_ls = {tf_q, tf_v1, tf_v2, tf_v3};
titles = {'Query', 'D1', 'D2', 'D3'};
tf_idf_dict = containers.Map();

figure;
for i = 1:length(tf_ls)
    tf_idf = containers.Map(terms, zeros(1, length(terms)));
    
    tf = tf_ls{i};
    for j = 1:length(terms)
        term = terms{j};
        if isKey(tf, term)
            tf_idf(term) = tf(term) * idf(j);
        end
    end
    
    tf_idf_dict(titles{i}) = tf_idf;
    
    tf_idf_values = zeros(1, length(terms));
    for k = 1:length(terms)
        tf_idf_values(k) = tf_idf(terms{k});
    end
    
    subplot(2, 2, i);
    bar(tf_idf_values, 'FaceColor', [0.2 0.4 0.6]);
    set(gca, 'xticklabel', terms);
    title(titles{i});
    ylabel('TF-IDF');
    xlabel('Terms');
end

fprintf('\nCosine sim in Document 1: %.4f\n', sim(tf_idf_dict('Query'), tf_idf_dict('D1'), terms));
fprintf('Cosine sim in Document 2: %.4f\n', sim(tf_idf_dict('Query'), tf_idf_dict('D2'), terms));
fprintf('Cosine sim in Document 3: %.4f\n', sim(tf_idf_dict('Query'), tf_idf_dict('D3'), terms));



function cos_sim = sim(query_tf_idf, doc_tf_idf, terms)
    numerator = 0;
    sum_query_sq = 0;
    sum_doc_sq = 0;
    
    for i = 1:length(terms)
        term = terms{i};
        query_val = query_tf_idf(term);
        doc_val = doc_tf_idf(term);
        numerator = numerator + query_val * doc_val;
        sum_query_sq = sum_query_sq + query_val^2;
        sum_doc_sq = sum_doc_sq + doc_val^2;
    end
    
    denominator = sqrt(sum_query_sq) * sqrt(sum_doc_sq);
    
    if denominator == 0
        cos_sim = 0;
    else
        cos_sim = numerator / denominator;
    end
end



function disp_map(map_obj)
    keys_list = keys(map_obj);
    values_list = values(map_obj);
    for i = 1:length(keys_list)
        key = keys_list{i};
        value = values_list{i};
        fprintf('%s: %.2f\n', key, value);
    end
end


function clean_doc = clean(document)
    document = lower(document);
    words = split(document);
    clean_doc = {};
    for i = 1:length(words)
        word = char(words{i});
        if all(isstrprop(word, 'alpha')) %is the word alphabetic
            clean_doc{end+1} = word;
        end
    end
end

function tf = build_tf(document)
    N = length(document);
    unique_words = unique(document);
    tf = containers.Map();
    for i = 1:length(unique_words)
        word = unique_words{i};
        word_count = sum(strcmp(document, word));
        tf(word) = round(word_count / N, 3);
    end
end