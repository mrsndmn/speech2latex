

mkdir -p ./speech2latex_code_submission/
cp -r .git ./speech2latex_code_submission/.git

cd ./speech2latex_code_submission/

# Восстанавливаем только закоммиченные файлы
git checkout .

# Удаляем все файлы, которые могут задеанонить (джобы, гит, скрипты c полными путями)
# Скрипты джобов для воспроизводимости не нужны и из них нельзя просто выкинуть полные пути (хотя можно, но костыльно это будет и неудобно)
rm -rf .git/ prepare_submission.sh ASRDataCreator ASR_FT RusTTS EngTTS envs

echo "Following files will be deleted:"
grep -Rl tarasov .
echo "Press Enter to continue"
read -n 1 -s
grep -Rl tarasov . | xargs rm

cd ..

# Create new archive
rm -rf speech2latex_code_submission.zip
zip -r speech2latex_code_submission.zip ./speech2latex_code_submission/
