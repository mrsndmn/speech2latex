

mkdir -p ./submission_code/
cp -r .git ./submission_code/.git

cd ./submission_code/

# Восстанавливаем только закоммиченные файлы
git checkout .

# Удаляем все файлы, которые могут задеанонить (джобы, гит, скрипты c полными путями)
# Скрипты джобов для воспроизводимости не нужны и из них нельзя просто выкинуть полные пути (хотя можно, но костыльно это будет и неудобно)
rm -rf .git/ prepare_submission.sh

echo "Following files will be deleted:"
grep -Rl tarasov .
echo "Press Enter to continue"
read -n 1 -s

grep -Rl tarasov . | xargs rm

cd ..

# Create new archive
rm -rf submission_code.zip
zip -r submission_code.zip ./submission_code/
